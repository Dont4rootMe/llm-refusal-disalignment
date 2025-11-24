"""
Unified training + generation + validation pipeline with Hydra support.

Flow:
 1) Load prompts for training (harmful/safe) via config.
 2) Fit steering method (currently SOM-based multi-directional ablation).
 3) Generate samples on eval prompts (harmful + safe) with the steered model.
 4) Run validation metrics (perplexity + refusal classification).

Run with Hydra (recommended):
  python -m pipeline method.name=som_md_ablation model_id=Qwen/Qwen3-VL-2B-Instruct
      train_data.source=synthetic train_data.n_pairs=200 eval_data.n_pairs=64

Hydra will auto-create an outputs directory; use output_dir to control where results are written.
Falls back to argparse if hydra is unavailable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Any, Dict

import torch

from algos.som_md_ablation import SOMAblationConfig, get_probed_model
from synthetic_data import SyntheticDataGenerator
from evaluation import metrics as eval_metrics

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from hydra.utils import get_original_cwd
    HYDRA_AVAILABLE = True
except Exception:
    HYDRA_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #


@dataclass
class DataConfig:
    source: str = "synthetic"  # "synthetic" or "files"
    n_pairs: int = 200  # used when source == "synthetic"
    seed: int = 123
    batch_size: int = 8
    safe_path: Optional[str] = None  # text file (one prompt per line) when source == "files"
    harmful_path: Optional[str] = None  # text file (one prompt per line) when source == "files"
    max_batches: Optional[int] = None
    max_items: Optional[int] = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    do_sample: bool = False


@dataclass
class MetricsConfig:
    ppl_model_id: Optional[str] = None  # default: same as pipeline model if None
    ppl_device: str = "cpu"
    classifier_model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    classifier_device: str = "cpu"
    classifier_batch_size: int = 4
    classifier_max_new_tokens: int = 32
    classifier_temperature: float = 0.0


@dataclass
class MethodConfig:
    name: str = "som_md_ablation"
    som_md_ablation: SOMAblationConfig = field(default_factory=SOMAblationConfig)


@dataclass
class PipelineConfig:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    output_dir: str = "outputs/pipeline"
    method: MethodConfig = field(default_factory=MethodConfig)
    train_data: DataConfig = field(default_factory=DataConfig)
    eval_data: DataConfig = field(default_factory=lambda: DataConfig(n_pairs=64))
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    seed: int = 42


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_prompts_from_file(path: str) -> List[str]:
    full = Path(path)
    if not full.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    texts: List[str] = []
    if full.suffix.lower() in {".json", ".jsonl"}:
        if full.suffix.lower() == ".jsonl":
            with full.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, str):
                            texts.append(obj)
                        elif isinstance(obj, dict):
                            if "prompt" in obj:
                                texts.append(str(obj["prompt"]))
                            elif "gen_text" in obj:
                                texts.append(str(obj["gen_text"]))
                    except Exception:
                        continue
        else:
            payload = json.loads(full.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict):
                        if "prompt" in item:
                            texts.append(str(item["prompt"]))
                        elif "gen_text" in item:
                            texts.append(str(item["gen_text"]))
            elif isinstance(payload, dict):
                for val in payload.values():
                    if isinstance(val, str):
                        texts.append(val)
                    elif isinstance(val, dict) and "gen_text" in val:
                        texts.append(str(val["gen_text"]))
    else:
        with full.open("r", encoding="utf-8") as fh:
            texts = [line.strip() for line in fh if line.strip()]
    if not texts:
        raise ValueError(f"No prompts loaded from {path}")
    return texts


def _prepare_prompts(cfg: DataConfig) -> Tuple[List[str], List[str]]:
    if cfg.source == "synthetic":
        gen = SyntheticDataGenerator(seed=cfg.seed)
        harmful, safe = gen.generate_pairs(n_pairs=cfg.n_pairs)
    elif cfg.source == "files":
        if not cfg.safe_path or not cfg.harmful_path:
            raise ValueError("files source requires safe_path and harmful_path")
        harmful = _load_prompts_from_file(cfg.harmful_path)
        safe = _load_prompts_from_file(cfg.safe_path)
    else:
        raise ValueError(f"Unknown data source: {cfg.source}")
    if cfg.max_items is not None:
        harmful = harmful[: cfg.max_items]
        safe = safe[: cfg.max_items]
    return harmful, safe


def _make_lazy_loader(texts: List[str], batch_size: int) -> Iterable[List[str]]:
    def gen():
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]
    return gen()


def _ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _generate_records(wrapper, prompts: List[str], is_refusal: bool, gen_cfg: GenerationConfig, start_idx: int = 0) -> List[dict]:
    records: List[dict] = []
    idx = start_idx
    for p in prompts:
        text = wrapper.generate_text(
            p,
            max_new_tokens=gen_cfg.max_new_tokens,
            temperature=gen_cfg.temperature,
            do_sample=gen_cfg.do_sample,
        )
        records.append({"object_index": idx, "ground_truth_is": is_refusal, "gen_text": text})
        idx += 1
    return records


# --------------------------------------------------------------------------- #
# Pipeline core
# --------------------------------------------------------------------------- #


def run_pipeline(cfg: PipelineConfig) -> dict:
    _set_seed(cfg.seed)

    try:
        base_dir = Path(get_original_cwd()) / cfg.output_dir if HYDRA_AVAILABLE else Path(cfg.output_dir)
    except Exception:
        base_dir = Path(cfg.output_dir)
    _ensure_output_dir(base_dir)

    logger.info("Preparing training data...")
    train_harmful, train_safe = _prepare_prompts(cfg.train_data)
    train_harmful_loader = _make_lazy_loader(train_harmful, cfg.train_data.batch_size)
    train_safe_loader = _make_lazy_loader(train_safe, cfg.train_data.batch_size)

    # Fit method
    if cfg.method.name != "som_md_ablation":
        raise NotImplementedError(f"Unsupported method: {cfg.method.name}")
    method_cfg = cfg.method.som_md_ablation
    method_cfg.default_gen_kwargs = {
        "max_new_tokens": cfg.generation.max_new_tokens,
        "temperature": cfg.generation.temperature,
        "do_sample": cfg.generation.do_sample,
    }

    logger.info("Fitting SOM-based multi-directional ablation...")
    steered = get_probed_model(method_cfg, cfg.model_id, safe_loader=train_safe_loader, harmful_loader=train_harmful_loader)

    # Eval prompts
    logger.info("Preparing eval data...")
    eval_harmful, eval_safe = _prepare_prompts(cfg.eval_data)

    logger.info("Generating samples on eval prompts...")
    records: List[dict] = []
    records.extend(_generate_records(steered, eval_harmful, is_refusal=True, gen_cfg=cfg.generation, start_idx=0))
    records.extend(_generate_records(steered, eval_safe, is_refusal=False, gen_cfg=cfg.generation, start_idx=len(records)))

    dataset_path = base_dir / "generated_records.json"
    with dataset_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    logger.info("Saved generated records to %s", dataset_path)

    metrics_cfg = cfg.metrics
    ppl_model = metrics_cfg.ppl_model_id or cfg.model_id
    report = eval_metrics.run_metrics(
        dataset_path=dataset_path,
        ppl_model_id=ppl_model,
        ppl_device=metrics_cfg.ppl_device,
        classifier_model_id=metrics_cfg.classifier_model_id,
        classifier_device=metrics_cfg.classifier_device,
        batch_size=metrics_cfg.classifier_batch_size,
        max_new_tokens=metrics_cfg.classifier_max_new_tokens,
        temperature=metrics_cfg.classifier_temperature,
    )

    report_path = base_dir / "metrics.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    logger.info("Saved metrics to %s", report_path)
    return report


# --------------------------------------------------------------------------- #
# Entrypoints
# --------------------------------------------------------------------------- #


def _register_hydra_configs():
    cs = ConfigStore.instance()
    cs.store(name="pipeline_config", node=PipelineConfig)


if HYDRA_AVAILABLE:
    _register_hydra_configs()


def _hydra_main(cfg: PipelineConfig) -> None:
    report = run_pipeline(cfg)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _argparse_main():
    parser = argparse.ArgumentParser(description="Run refusal pipeline (Hydra recommended; argparse is fallback)")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct", help="HF model id")
    parser.add_argument("--output-dir", default="outputs/pipeline", help="Where to save outputs")
    parser.add_argument("--train-n", type=int, default=200, help="Number of synthetic pairs for training")
    parser.add_argument("--eval-n", type=int, default=64, help="Number of synthetic pairs for eval")
    parser.add_argument("--device", default="cpu", help="Device string for steering method (e.g., cuda:0)")
    args = parser.parse_args()

    cfg = PipelineConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        train_data=DataConfig(n_pairs=args.train_n, source="synthetic"),
        eval_data=DataConfig(n_pairs=args.eval_n, source="synthetic"),
    )
    cfg.method.som_md_ablation.device = args.device
    report = run_pipeline(cfg)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if HYDRA_AVAILABLE:
        hydra.main(config_name="pipeline_config", version_base="1.3")(_hydra_main)()
    else:
        _argparse_main()
