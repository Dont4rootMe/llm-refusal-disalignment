"""
Unified training + generation + validation pipeline with Hydra support.

Flow:
 1) Collect prompts from synthetic templates (configurable per template/word), real datasets, and/or manual files.
 2) Split into train/validation via ratio; limit counts per split if needed.
 3) Fit steering method (som_md_ablation or refusal_reduction) on the train split.
 4) Generate responses on the validation split and run metrics (perplexity + refusal classification).

Run with Hydra (recommended):
  python -m pipeline --config-dir conf --config-name pipeline
      method.name=som_md_ablation data.synthetic.mode=selection_map data.real.enabled=true

Hydra will auto-create an outputs directory; use output_dir to control where results are written.
Falls back to argparse if hydra is unavailable.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch

from algos.original_refusal_reduction import (
    RefusalDirectionExtractor,
    RefusalReductionConfig,
    RefusalReductionModel,
)
from algos.som_md_ablation import SOMAblationConfig, get_probed_model
from evaluation import metrics as eval_metrics
from synthetic_data import SyntheticDataGenerator

try:
    from data.real_data import CorpusBuilder
except Exception:
    CorpusBuilder = None

try:
    import hydra
    from hydra.core.config_store import ConfigStore
    from hydra.utils import get_original_cwd
    from omegaconf import OmegaConf

    HYDRA_AVAILABLE = True
except Exception:
    HYDRA_AVAILABLE = False
    OmegaConf = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #


@dataclass
class SyntheticConfig:
    enabled: bool = True
    mode: str = "pairs"  # "pairs" | "selection_map"
    n_pairs: int = 200
    variant_frac: float = 0.3
    selection_map: Dict[str, Any] = field(default_factory=dict)  # templates/toxic/benign as int or list[int]
    backtranslation_passes: int = 0
    paraphrase_variants: int = 1
    seed: int = 123


@dataclass
class RealDataConfig:
    enabled: bool = False
    data_dir: str = "./datasets"
    default_count: int = 20
    selection_counts: Dict[str, int] = field(default_factory=lambda: {"attaq": 0, "gandalf": 0, "wildguard": 0})


@dataclass
class ManualFileConfig:
    enabled: bool = False
    safe_path: Optional[str] = None
    harmful_path: Optional[str] = None
    max_items: Optional[int] = None


@dataclass
class DataMixConfig:
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    real: RealDataConfig = field(default_factory=RealDataConfig)
    manual: ManualFileConfig = field(default_factory=ManualFileConfig)
    validation_ratio: float = 0.25
    batch_size: int = 8
    max_train_items: Optional[int] = None
    max_eval_items: Optional[int] = None
    shuffle_seed: int = 42


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
    refusal_reduction: RefusalReductionConfig = field(default_factory=RefusalReductionConfig)


@dataclass
class PipelineConfig:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    output_dir: str = "outputs/pipeline"
    method: MethodConfig = field(default_factory=MethodConfig)
    data: DataMixConfig = field(default_factory=DataMixConfig)
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


def _prepare_synthetic_prompts(cfg: SyntheticConfig, global_seed: int) -> Tuple[List[str], List[str]]:
    if not cfg.enabled:
        return [], []
    seed = cfg.seed if cfg.seed is not None else global_seed
    gen = SyntheticDataGenerator(seed=seed)
    if cfg.mode == "selection_map":
        selection_map = dict(cfg.selection_map or {})

        def _normalize(value, total):
            if isinstance(value, str) and value.lower() == "all":
                return list(range(total))
            if value is None:
                return list(range(total))
            return value

        selection_map["templates"] = _normalize(selection_map.get("templates"), len(gen.template_pool))
        selection_map["toxic"] = _normalize(selection_map.get("toxic"), len(gen.toxic_words))
        selection_map["benign"] = _normalize(selection_map.get("benign"), len(gen.benign_words))
        harmful_ds, safe_ds = gen.generate_by_selection_map(selection_map)
        harmful = list(getattr(harmful_ds, "texts", list(harmful_ds)))
        safe = list(getattr(safe_ds, "texts", list(safe_ds)))
    else:
        harmful, safe = gen.generate_pairs(n_pairs=cfg.n_pairs, variant_frac=cfg.variant_frac)

    if cfg.backtranslation_passes > 0:
        harmful = gen.augment_with_backtranslation(harmful, n_passes=cfg.backtranslation_passes)
        safe = gen.augment_with_backtranslation(safe, n_passes=cfg.backtranslation_passes)
    if cfg.paraphrase_variants and cfg.paraphrase_variants > 1:
        harmful = gen.neural_paraphrase_expand(harmful, n_variants=cfg.paraphrase_variants)
        safe = gen.neural_paraphrase_expand(safe, n_variants=cfg.paraphrase_variants)
    return harmful, safe


def _prepare_real_prompts(cfg: RealDataConfig) -> Tuple[List[str], List[str]]:
    if not cfg.enabled:
        return [], []
    if CorpusBuilder is None:
        raise RuntimeError("Real data loading requires the `datasets` package; install it to enable data.real_data.")
    try:
        builder = CorpusBuilder(cfg.data_dir)
    except Exception as exc:  # pragma: no cover - handled with message
        raise RuntimeError("Failed to initialize CorpusBuilder (datasets dependency missing?)") from exc

    selection = {k: v for k, v in (cfg.selection_counts or {}).items() if v and v > 0}
    selection = selection or None
    dataset = builder.generate_dataset(selection_counts=selection, default_count=cfg.default_count)
    harmful = [row["text"] for row in dataset if int(row["label"]) == 1]
    safe = [row["text"] for row in dataset if int(row["label"]) == 0]
    return harmful, safe


def _prepare_manual_prompts(cfg: ManualFileConfig) -> Tuple[List[str], List[str]]:
    if not cfg.enabled:
        return [], []
    if not cfg.safe_path or not cfg.harmful_path:
        raise ValueError("manual files source requires both safe_path and harmful_path")
    harmful = _load_prompts_from_file(cfg.harmful_path)
    safe = _load_prompts_from_file(cfg.safe_path)
    if cfg.max_items is not None:
        harmful = harmful[: cfg.max_items]
        safe = safe[: cfg.max_items]
    return harmful, safe


def _collect_prompts(data_cfg: DataMixConfig, global_seed: int) -> Tuple[List[str], List[str]]:
    harmful: List[str] = []
    safe: List[str] = []
    h_syn, s_syn = _prepare_synthetic_prompts(data_cfg.synthetic, global_seed)
    harmful.extend(h_syn)
    safe.extend(s_syn)

    h_real, s_real = _prepare_real_prompts(data_cfg.real)
    harmful.extend(h_real)
    safe.extend(s_real)

    h_manual, s_manual = _prepare_manual_prompts(data_cfg.manual)
    harmful.extend(h_manual)
    safe.extend(s_manual)

    if not harmful or not safe:
        raise ValueError("No data collected from any source (synthetic/real/manual)")
    return harmful, safe


def _split_train_eval(harmful: List[str], safe: List[str], ratio: float, seed: int) -> Tuple[List[str], List[str], List[str], List[str]]:
    if not 0 <= ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    rnd = random.Random(seed)

    def split(items: List[str]) -> Tuple[List[str], List[str]]:
        if not items:
            return [], []
        idxs = list(range(len(items)))
        rnd.shuffle(idxs)
        cut = int(len(items) * ratio)
        eval_idx = set(idxs[:cut])
        train = [items[i] for i in range(len(items)) if i not in eval_idx]
        eval_set = [items[i] for i in range(len(items)) if i in eval_idx]
        return train, eval_set

    train_h, eval_h = split(harmful)
    train_s, eval_s = split(safe)
    if not train_h or not train_s:
        raise ValueError("Train split is empty; reduce validation_ratio or provide more data.")
    return train_h, train_s, eval_h, eval_s


def _limit_list(items: List[str], limit: Optional[int]) -> List[str]:
    if limit is None:
        return items
    return items[: max(0, limit)]


def _make_lazy_loader(texts: List[str], batch_size: int) -> Iterable[List[str]]:
    def gen():
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]
    return gen()


def _merge_dict_into_dataclass(dc_obj, updates: Mapping[str, Any]):
    for field_info in fields(dc_obj):
        name = field_info.name
        if name not in updates:
            continue
        current = getattr(dc_obj, name)
        incoming = updates[name]
        if is_dataclass(current):
            merged = _merge_dict_into_dataclass(copy.deepcopy(current), incoming or {})
            setattr(dc_obj, name, merged)
        else:
            setattr(dc_obj, name, incoming)
    return dc_obj


def _coerce_pipeline_config(cfg: Any) -> PipelineConfig:
    if isinstance(cfg, PipelineConfig):
        return cfg
    if HYDRA_AVAILABLE and OmegaConf is not None:
        try:
            base = OmegaConf.structured(PipelineConfig)
            merged = OmegaConf.merge(base, cfg)
            return OmegaConf.to_object(merged)  # type: ignore
        except Exception:
            logger.exception("Failed to coerce Hydra config; falling back to manual merge.")
    if isinstance(cfg, Mapping):
        base = copy.deepcopy(PipelineConfig())
        return _merge_dict_into_dataclass(base, cfg)
    raise TypeError(f"Unsupported config type: {type(cfg)}")


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


class _RefusalReductionWrapper:
    """Adapter to expose RefusalReductionModel as generate_text-compatible."""

    def __init__(self, model: RefusalReductionModel, gen_cfg: GenerationConfig):
        self.model = model
        self.default_kwargs = {
            "max_new_tokens": gen_cfg.max_new_tokens,
            "temperature": gen_cfg.temperature,
            "do_sample": gen_cfg.do_sample,
        }
        # Attach hooks once; further generate calls keep them active.
        try:
            self.model._setup_intervention_hooks()
        except Exception:
            logger.warning("Could not register intervention hooks for refusal reduction model.")

    def generate_text(self, prompt: str, **gen_kwargs) -> str:
        kwargs = dict(self.default_kwargs)
        kwargs.update(gen_kwargs)
        return self.model._generate(prompt, **kwargs)

    def cleanup(self):
        try:
            self.model.remove_intervention_hooks()
        except Exception:
            pass


def _build_method_wrapper(cfg: PipelineConfig, train_safe: List[str], train_harmful: List[str]) -> Any:
    if cfg.method.name == "som_md_ablation":
        method_cfg = cfg.method.som_md_ablation
        method_cfg.default_gen_kwargs = {
            "max_new_tokens": cfg.generation.max_new_tokens,
            "temperature": cfg.generation.temperature,
            "do_sample": cfg.generation.do_sample,
        }
        safe_loader = _make_lazy_loader(train_safe, cfg.data.batch_size)
        harmful_loader = _make_lazy_loader(train_harmful, cfg.data.batch_size)
        logger.info("Fitting SOM-based multi-directional ablation...")
        return get_probed_model(method_cfg, cfg.model_id, safe_loader=safe_loader, harmful_loader=harmful_loader)

    if cfg.method.name == "refusal_reduction":
        rr_cfg = cfg.method.refusal_reduction
        max_per_class = rr_cfg.max_samples_per_class
        safe_samples = train_safe[:max_per_class] if max_per_class else train_safe
        harmful_samples = train_harmful[:max_per_class] if max_per_class else train_harmful
        logger.info("Computing refusal directions (PCA-style) for reduction model...")
        extractor = RefusalDirectionExtractor(cfg.model_id, config=rr_cfg)
        directions = extractor.compute_refusal_directions(harmful_samples, safe_samples, k=rr_cfg.k_pca)
        reduction_model = RefusalReductionModel(cfg.model_id, directions, config=rr_cfg)
        return _RefusalReductionWrapper(reduction_model, cfg.generation)

    raise NotImplementedError(f"Unsupported method: {cfg.method.name}")


# --------------------------------------------------------------------------- #
# Pipeline core
# --------------------------------------------------------------------------- #


def run_pipeline(cfg: PipelineConfig) -> dict:
    cfg = _coerce_pipeline_config(cfg)
    _set_seed(cfg.seed)

    try:
        base_dir = Path(get_original_cwd()) / cfg.output_dir if HYDRA_AVAILABLE else Path(cfg.output_dir)
    except Exception:
        base_dir = Path(cfg.output_dir)
    _ensure_output_dir(base_dir)

    logger.info("Collecting data from synthetic/real/manual sources...")
    harmful_all, safe_all = _collect_prompts(cfg.data, cfg.seed)

    train_harmful, train_safe, eval_harmful, eval_safe = _split_train_eval(
        harmful_all,
        safe_all,
        cfg.data.validation_ratio,
        cfg.data.shuffle_seed,
    )
    train_harmful = _limit_list(train_harmful, cfg.data.max_train_items)
    train_safe = _limit_list(train_safe, cfg.data.max_train_items)
    eval_harmful = _limit_list(eval_harmful, cfg.data.max_eval_items)
    eval_safe = _limit_list(eval_safe, cfg.data.max_eval_items)

    if not train_harmful or not train_safe:
        raise ValueError("Training split is empty after applying limits; provide more data or relax limits.")

    if not eval_harmful and not eval_safe:
        logger.warning("Validation set is empty; falling back to training data for validation.")
        eval_harmful, eval_safe = train_harmful, train_safe

    logger.info(
        "Data split -> train_harmful=%d, train_safe=%d, eval_harmful=%d, eval_safe=%d",
        len(train_harmful),
        len(train_safe),
        len(eval_harmful),
        len(eval_safe),
    )

    split_report = {
        "train_harmful": len(train_harmful),
        "train_safe": len(train_safe),
        "eval_harmful": len(eval_harmful),
        "eval_safe": len(eval_safe),
    }
    with (base_dir / "data_split.json").open("w", encoding="utf-8") as fh:
        json.dump(split_report, fh, indent=2, ensure_ascii=False)

    logger.info("Fitting method: %s", cfg.method.name)
    model_wrapper = _build_method_wrapper(cfg, train_safe=train_safe, train_harmful=train_harmful)

    logger.info("Generating samples on validation prompts...")
    records: List[dict] = []
    records.extend(_generate_records(model_wrapper, eval_harmful, is_refusal=True, gen_cfg=cfg.generation, start_idx=0))
    records.extend(_generate_records(model_wrapper, eval_safe, is_refusal=False, gen_cfg=cfg.generation, start_idx=len(records)))

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
    if hasattr(model_wrapper, "cleanup"):
        try:
            model_wrapper.cleanup()
        except Exception:
            pass

    report["data_split"] = split_report
    report["generated_path"] = str(dataset_path)
    report["metrics_path"] = str(report_path)
    return report


# --------------------------------------------------------------------------- #
# Entrypoints
# --------------------------------------------------------------------------- #


def _register_hydra_configs():
    cs = ConfigStore.instance()
    cs.store(name="pipeline", node=PipelineConfig)


if HYDRA_AVAILABLE:
    _register_hydra_configs()


def _hydra_main(cfg: PipelineConfig) -> None:
    report = run_pipeline(cfg)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _argparse_main():
    parser = argparse.ArgumentParser(description="Run refusal pipeline (Hydra recommended; argparse is fallback)")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct", help="HF model id")
    parser.add_argument("--output-dir", default="outputs/pipeline", help="Where to save outputs")
    parser.add_argument("--method", choices=["som_md_ablation", "refusal_reduction"], default="som_md_ablation")
    parser.add_argument("--synthetic-n", type=int, default=200, help="How many synthetic pairs to generate")
    parser.add_argument("--validation-ratio", type=float, default=0.25, help="Fraction of data to use for validation")
    parser.add_argument("--device", default="cpu", help="Device string for steering method (e.g., cuda:0)")
    parser.add_argument("--real-attaq", type=int, default=0, help="How many AttaQ samples to pull (0 disables real data)")
    parser.add_argument("--real-gandalf", type=int, default=0, help="How many Gandalf samples to pull")
    parser.add_argument("--real-wildguard", type=int, default=0, help="How many WildGuard samples to pull (per class)")
    args = parser.parse_args()

    cfg = PipelineConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
    )
    cfg.method.name = args.method
    cfg.data.validation_ratio = args.validation_ratio
    cfg.data.synthetic.n_pairs = args.synthetic_n
    cfg.data.synthetic.enabled = True

    real_counts = {"attaq": args.real_attaq, "gandalf": args.real_gandalf, "wildguard": args.real_wildguard}
    cfg.data.real.selection_counts = real_counts
    cfg.data.real.enabled = any(v > 0 for v in real_counts.values())
    cfg.method.som_md_ablation.device = args.device
    cfg.method.refusal_reduction.activation_device = args.device
    report = run_pipeline(cfg)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if HYDRA_AVAILABLE:
        hydra.main(config_path="conf", config_name="pipeline", version_base="1.3")(_hydra_main)()
    else:
        _argparse_main()
