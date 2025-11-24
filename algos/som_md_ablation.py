"""
SOM-based multi-directional refusal ablation (Algorithm 1 from arXiv:2511.08379v2).

Implements the pipeline:
 1) Extract hidden states at a target layer for harmful and harmless prompts.
 2) Compute harmless centroid.
 3) Train a Self-Organizing Map (SOM) over harmful representations.
 4) Derive multiple refusal directions from SOM neurons minus centroid.
 5) Optionally run a small Bayesian-style random search over direction subsets.
 6) Attach projection-ablation hooks to the model so `.generate(...)` uses the steered behavior.

Entrypoint: `get_probed_model(config, model_name, safe_loader, harmful_loader)`
Returns a model wrapper with hooks applied (and attached tokenizer for convenience).
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union, Any, Dict

import torch
import torch.nn as nn

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:  # pragma: no cover - dependency should exist in the env
    raise RuntimeError("transformers package is required for som_md_ablation") from e

# Optional judge model (falls back to heuristics if unavailable)
try:
    from evaluation.response_classification import ResponseClassificator  # type: ignore
except Exception:
    ResponseClassificator = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class SOMAblationConfig:
    # Targeting / extraction
    target_layer: Optional[str] = None
    max_seq_len: int = 1024
    device: str = "cpu"
    max_harmful_batches: Optional[int] = None
    max_safe_batches: Optional[int] = None
    max_harmful_items: Optional[int] = 4000
    max_safe_items: Optional[int] = 6000

    # SOM hyperparameters
    som_rows: int = 4
    som_cols: int = 4
    som_steps: int = 5000
    som_lr0: float = 0.1
    som_sigma0: float = 0.8
    som_seed: int = 42

    # Direction selection
    k_directions: int = 3
    normalize_directions: bool = True
    bo_trials: int = 24  # random-search approximation of BO
    bo_eval_harmful: int = 24  # how many harmful prompts to judge per trial
    bo_temperature: float = 0.0
    bo_max_new_tokens: int = 96
    bo_classifier_model: Optional[str] = None  # use ResponseClassificator if available

    # Ablation application
    ablation_strength: float = 1.0
    default_gen_kwargs: Dict[str, Any] = field(default_factory=lambda: {"max_new_tokens": 128, "temperature": 0.0})

    # Misc
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _set_default_dtype_kwargs(config: SOMAblationConfig, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(kwargs)
    if torch.cuda.is_available() and config.device.startswith("cuda") and "torch_dtype" not in out:
        out["torch_dtype"] = torch.float16
    if torch.cuda.is_available() and "device_map" not in out and config.device.startswith("cuda"):
        out["device_map"] = "auto"
    return out


def _load_model_and_tokenizer(model_name: str, config: SOMAblationConfig):
    kwargs = _set_default_dtype_kwargs(config, config.model_kwargs)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **kwargs)
    try:
        if kwargs.get("device_map") is None:
            model.to(torch.device(config.device))
    except Exception:
        logger.warning("Could not move model to device %s", config.device)
    model.eval()
    return model, tok


def _guess_target_module(model: nn.Module, target_layer: Optional[str]) -> Tuple[str, nn.Module]:
    if target_layer:
        for name, module in model.named_modules():
            if name == target_layer or name.endswith(target_layer):
                return name, module
        logger.warning("Target layer '%s' not found; falling back to heuristic last block", target_layer)
    candidate = None
    candidate_name = ""
    max_idx = -1
    for name, module in model.named_modules():
        if any(k in name.lower() for k in ("layers", "h.", "block")):
            candidate = module
            candidate_name = name
            parts = name.replace("layers", "").replace("block", "").replace("h", "").split(".")
            nums = [int(p) for p in parts if p.isdigit()]
            if nums:
                idx = max(nums)
                if idx > max_idx:
                    max_idx = idx
                    candidate = module
                    candidate_name = name
    if candidate is None:
        raise RuntimeError("Could not infer a target layer; please provide config.target_layer")
    return candidate_name, candidate


def _flatten_text_batch(batch: Union[str, List[Any], Tuple[Any, ...]]) -> List[str]:
    if isinstance(batch, str):
        return [batch]
    if isinstance(batch, (list, tuple)):
        out: List[str] = []
        for item in batch:
            out.extend(_flatten_text_batch(item))
        return [str(x) for x in out]
    return [str(batch)]


def _materialize_loader(loader: Iterable, max_batches: Optional[int]) -> List[Any]:
    """
    Copy batches from loader into a list so we can iterate multiple times (needed for BO search).
    """
    if isinstance(loader, list):
        return loader if max_batches is None else loader[:max_batches]
    out: List[Any] = []
    for idx, batch in enumerate(loader):
        if max_batches is not None and idx >= max_batches:
            break
        out.append(batch)
    return out


def _collect_hidden_reps(
    model: nn.Module,
    tokenizer,
    loader: Iterable,
    target_module: nn.Module,
    max_batches: Optional[int],
    max_items: Optional[int],
    max_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    collected: List[torch.Tensor] = []
    seen = 0

    def hook(_module, _inp, output):
        nonlocal seen
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden is None or hidden.dim() != 3:
            return output
        last_tok = hidden[:, -1, :].detach().to("cpu")
        collected.append(last_tok)
        seen += last_tok.shape[0]
        return output

    handle = target_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for b_idx, batch in enumerate(loader):
                if max_batches is not None and b_idx >= max_batches:
                    break
                if max_items is not None and seen >= max_items:
                    break
                texts = _flatten_text_batch(batch)
                enc = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_seq_len,
                ).to(device)
                model(**enc)
                if max_items is not None and seen >= max_items:
                    break
    finally:
        handle.remove()

    if not collected:
        raise RuntimeError("No hidden states were collected; check loaders and target layer.")
    return torch.cat(collected, dim=0)


# --------------------------------------------------------------------------- #
# Self-Organizing Map
# --------------------------------------------------------------------------- #

class SimpleSOM:
    """Minimal SOM with Gaussian neighborhood on a 2D grid."""

    def __init__(
        self,
        rows: int,
        cols: int,
        dim: int,
        steps: int = 5000,
        lr0: float = 0.1,
        sigma0: float = 0.8,
        seed: int = 42,
        device: Optional[torch.device] = None,
    ):
        self.rows = int(rows)
        self.cols = int(cols)
        self.dim = int(dim)
        self.steps = int(steps)
        self.lr0 = float(lr0)
        self.sigma0 = float(sigma0)
        self.device = device or torch.device("cpu")
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        self.positions = self._make_positions().to(self.device)
        self.weights = torch.randn(self.rows * self.cols, self.dim, generator=g, device=self.device) * 0.01

    def _make_positions(self) -> torch.Tensor:
        coords = []
        for r in range(self.rows):
            for c in range(self.cols):
                coords.append((float(r), float(c)))
        return torch.tensor(coords, dtype=torch.float32)

    def _neighborhood(self, bmu_idx: int, sigma: float) -> torch.Tensor:
        pos_bmu = self.positions[bmu_idx]
        dist2 = torch.sum((self.positions - pos_bmu) ** 2, dim=1)
        return torch.exp(-dist2 / (2.0 * sigma * sigma))

    def fit(self, data: torch.Tensor) -> torch.Tensor:
        if data.dim() != 2:
            raise ValueError("SOM input must be 2D (n, dim)")
        data = data.to(self.device)
        n = data.shape[0]
        for t in range(self.steps):
            idx = random.randint(0, n - 1)
            x = data[idx]
            dists = torch.sum((self.weights - x) ** 2, dim=1)
            bmu_idx = int(torch.argmin(dists).item())
            lr = self.lr0 / (1.0 + 2.0 * t / max(1, self.steps))
            sigma = max(1e-3, self.sigma0 * (1.0 - t / max(1, self.steps)))
            neigh = self._neighborhood(bmu_idx, sigma)  # (num_neurons,)
            delta = lr * neigh.unsqueeze(1) * (x.unsqueeze(0) - self.weights)
            self.weights = self.weights + delta
        return self.weights.detach().cpu()


# --------------------------------------------------------------------------- #
# Hooked ablation wrapper
# --------------------------------------------------------------------------- #

class AblationHook:
    def __init__(self, directions: torch.Tensor, strength: float):
        """
        directions: (k, hidden) normalized or not
        """
        if directions.dim() == 1:
            directions = directions.unsqueeze(0)
        self.directions = directions  # kept on CPU, moved per forward call
        self.strength = float(strength)

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden is None or hidden.dim() != 3:
            return output
        dirs = self.directions.to(hidden.device)
        flat = hidden.reshape(-1, hidden.shape[-1])
        proj = flat @ dirs.T  # (bs*seq, k)
        recon = proj @ dirs  # (bs*seq, hidden)
        mod_flat = flat - self.strength * recon
        mod = mod_flat.reshape_as(hidden)
        if isinstance(output, (tuple, list)):
            return (mod,) + tuple(output[1:])
        return mod


class SteeredModelWrapper:
    """
    Lightweight wrapper exposing:
      - .model (underlying huggingface model)
      - .tokenizer
      - .generate(...) passthrough
      - .generate_text(prompt: str, **kwargs) convenience
    """

    def __init__(self, model, tokenizer, hook_handle, default_gen_kwargs: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self._hook_handle = hook_handle
        self.default_gen_kwargs = default_gen_kwargs or {}

    def remove_hooks(self):
        try:
            if self._hook_handle is not None:
                self._hook_handle.remove()
        except Exception:
            pass

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def generate_text(self, prompt: str, **gen_kwargs) -> str:
        kwargs = dict(self.default_gen_kwargs)
        kwargs.update(gen_kwargs)
        device = next(self.model.parameters()).device
        enc = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).to(device)
        with torch.no_grad():
            out = self.model.generate(**enc, **kwargs)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        return text


# --------------------------------------------------------------------------- #
# Judge utilities
# --------------------------------------------------------------------------- #

_REFUSAL_KEYWORDS = (
    "sorry",
    "cannot",
    "can't",
    "unable",
    "not able",
    "as an ai",
    "i am just an ai",
    "i'm just an ai",
    "not allowed",
    "refuse",
    "decline",
)


def _heuristic_is_refusal(text: str) -> bool:
    tl = text.lower()
    return any(k in tl for k in _REFUSAL_KEYWORDS)


def _score_responses_as_compliance(responses: List[str]) -> float:
    if not responses:
        return 0.0
    non_refusal = sum(1 for r in responses if not _heuristic_is_refusal(r))
    return non_refusal / len(responses)


def _make_classifier_judge(model_id: str, device: str, batch_size: int, max_new_tokens: int, temperature: float):
    if ResponseClassificator is None:
        return None
    try:
        clf = ResponseClassificator(
            model_id=model_id,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return clf
    except Exception:
        logger.warning("Could not initialize ResponseClassificator judge; falling back to heuristics.")
        return None


# --------------------------------------------------------------------------- #
# Direction search
# --------------------------------------------------------------------------- #

def _evaluate_subset(
    model: nn.Module,
    tokenizer,
    target_module: nn.Module,
    directions: torch.Tensor,
    subset_indices: Sequence[int],
    eval_prompts: List[str],
    strength: float,
    gen_kwargs: Dict[str, Any],
) -> List[str]:
    if not subset_indices:
        return []
    subset_dirs = directions[subset_indices]
    hook = target_module.register_forward_hook(AblationHook(subset_dirs, strength))
    responses: List[str] = []
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            for p in eval_prompts:
                enc = tokenizer(
                    [p],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                ).to(device)
                out = model.generate(**enc, **gen_kwargs)
                text = tokenizer.decode(out[0], skip_special_tokens=True)
                if text.startswith(p):
                    text = text[len(p):].strip()
                responses.append(text)
    finally:
        hook.remove()
    return responses


def _select_directions_with_bo(
    model: nn.Module,
    tokenizer,
    target_module: nn.Module,
    directions: torch.Tensor,
    k: int,
    eval_prompts: List[str],
    config: SOMAblationConfig,
) -> List[int]:
    num_dirs = directions.shape[0]
    k = min(k, num_dirs)
    if k <= 0:
        return []
    gen_kwargs = dict(config.default_gen_kwargs)
    gen_kwargs.update({"temperature": config.bo_temperature, "max_new_tokens": config.bo_max_new_tokens})

    classifier = None
    if config.bo_classifier_model:
        classifier = _make_classifier_judge(
            config.bo_classifier_model, config.device, batch_size=2, max_new_tokens=24, temperature=0.0
        )

    best_score = -1.0
    best_subset: List[int] = list(range(k))
    rng = random.Random(config.som_seed + 17)

    for trial in range(max(1, config.bo_trials)):
        subset = rng.sample(range(num_dirs), k) if num_dirs >= k else list(range(num_dirs))
        responses = _evaluate_subset(model, tokenizer, target_module, directions, subset, eval_prompts, config.ablation_strength, gen_kwargs)

        if classifier is not None:
            try:
                preds = classifier.classify(responses)
                score = sum(1 for p in preds if isinstance(p, str) and p.strip().lower() != "refusal") / max(1, len(preds))
            except Exception:
                logger.warning("Classifier judge failed on trial %d; reverting to heuristic.", trial)
                score = _score_responses_as_compliance(responses)
        else:
            score = _score_responses_as_compliance(responses)

        if score > best_score:
            best_score = score
            best_subset = subset
        logger.debug("BO trial %d subset=%s score=%.3f", trial, subset, score)
    logger.info("Selected directions (k=%d) with score=%.3f: %s", k, best_score, best_subset)
    return list(best_subset)


# --------------------------------------------------------------------------- #
# Main entrypoint
# --------------------------------------------------------------------------- #

def get_probed_model(
    config: SOMAblationConfig,
    model_name: str,
    safe_loader: Iterable,
    harmful_loader: Iterable,
) -> SteeredModelWrapper:
    """
    Apply SOM-based multi-directional refusal ablation and return a model wrapper ready for generate(...).

    Args:
        config: SOMAblationConfig with hyperparameters.
        model_name: huggingface model id or local path.
        safe_loader: iterable yielding safe prompts (strings) batches.
        harmful_loader: iterable yielding harmful prompts (strings) batches.
    """
    device = torch.device(config.device)
    model, tokenizer = _load_model_and_tokenizer(model_name, config)
    target_name, target_module = _guess_target_module(model, config.target_layer)
    logger.info("Using target layer: %s", target_name)

    harmful_batches = _materialize_loader(harmful_loader, config.max_harmful_batches)
    safe_batches = _materialize_loader(safe_loader, config.max_safe_batches)

    logger.info("Collecting hidden representations (harmful)...")
    reps_hf = _collect_hidden_reps(
        model,
        tokenizer,
        iter(harmful_batches),
        target_module,
        max_batches=None,
        max_items=config.max_harmful_items,
        max_seq_len=config.max_seq_len,
        device=device,
    )
    logger.info("Collected harmful reps: %s", tuple(reps_hf.shape))

    logger.info("Collecting hidden representations (harmless)...")
    reps_hl = _collect_hidden_reps(
        model,
        tokenizer,
        iter(safe_batches),
        target_module,
        max_batches=None,
        max_items=config.max_safe_items,
        max_seq_len=config.max_seq_len,
        device=device,
    )
    logger.info("Collected harmless reps: %s", tuple(reps_hl.shape))

    centroid = reps_hl.mean(dim=0, keepdim=True)  # (1, hidden)

    logger.info("Training SOM on harmful representations...")
    som = SimpleSOM(
        rows=config.som_rows,
        cols=config.som_cols,
        dim=reps_hf.shape[1],
        steps=config.som_steps,
        lr0=config.som_lr0,
        sigma0=config.som_sigma0,
        seed=config.som_seed,
        device=device,
    )
    neurons = som.fit(reps_hf)  # (num_neurons, hidden)
    directions = neurons - centroid  # broadcast
    if config.normalize_directions:
        norms = directions.norm(dim=1, keepdim=True).clamp(min=1e-8)
        directions = directions / norms

    logger.info("Derived %d candidate directions from SOM", directions.shape[0])

    # Prepare evaluation prompts for selection
    eval_prompts: List[str] = []
    for b_idx, batch in enumerate(harmful_batches):
        eval_prompts.extend(_flatten_text_batch(batch))
        if len(eval_prompts) >= config.bo_eval_harmful:
            break
    eval_prompts = eval_prompts[: config.bo_eval_harmful]

    if directions.shape[0] <= config.k_directions or config.bo_trials <= 0:
        selected_idx = list(range(min(config.k_directions, directions.shape[0])))
    else:
        selected_idx = _select_directions_with_bo(
            model,
            tokenizer,
            target_module,
            directions,
            k=config.k_directions,
            eval_prompts=eval_prompts,
            config=config,
        )

    selected_dirs = directions[selected_idx]
    logger.info("Selected %d directions; registering ablation hook", selected_dirs.shape[0])

    handle = target_module.register_forward_hook(AblationHook(selected_dirs, config.ablation_strength))
    wrapper = SteeredModelWrapper(model, tokenizer, handle, config.default_gen_kwargs)
    return wrapper


__all__ = [
    "SOMAblationConfig",
    "get_probed_model",
]
