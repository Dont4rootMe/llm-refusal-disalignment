import os
import sys
import logging
from typing import Dict, Optional, Union
from datasets import load_from_disk, concatenate_datasets, Dataset, Value, Features

logger = logging.getLogger(__name__)


class CorpusBuilder:
    """
    Orchestrates the loading, normalization, and merging of heterogeneous safety datasets.
    Yields datasets with a strict schema: {'text': string, 'label': int64}.

    Supported Datasets:
    -------------------
    1. IBM AttaQ (ibm-research/AttaQ)
       - Source: https://huggingface.co/datasets/ibm-research/AttaQ
       - Type:   SINGLE class: Harmful
       - Size:   ~1.4k examples
       - Desc:   Collection of potential adversarial attacks on LLMs.

    2. Lakera Gandalf (Lakera/gandalf_ignore_instructions)
       - Source: https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions
       - Type:   SINGLE class: Harmful
       - Size:   ~1k examples
       - Desc:   Prompts designed to bypass instructions (ignore previous instructions).

    3. WildGuard (allenai/wildguardmix)
       - Source: https://huggingface.co/datasets/allenai/wildguardmix
       - Type:   MIXED (Safe & Harmful)
       - Size:   ~88k examples
       - Desc:   Large-scale dataset containing both benign user queries and
                 harmful jailbreak attempts/toxic prompts.

    Labels Schema:
    --------------
    0: Safe
    1: Harmful
    """

    DATASET_CONFIGS = {
        "attaq": {
            "folder_name": "attaq",
            "mode": "single_harmful",
            "text_col_source": "input",
        },
        "gandalf": {
            "folder_name": "gandalf",
            "mode": "single_harmful",
            "text_col_source": "text",
        },
        "wildguard": {
            "folder_name": "wildguard",
            "mode": "mixed",
            "text_col_source": "prompt",
            "filter_col": "prompt_harm_label",
            "filter_val_harmful": "harmful",
            "filter_val_safe": "unharmful"
        }
    }

    def __init__(self, data_dir: str = "./datasets"):
        """
        Args:
            data_dir: Root directory containing the subfolders for each dataset.
        """
        self.data_dir = data_dir

    def _load_dataset(self, key: str) -> Optional[Dataset]:
        """Loads a specific dataset from disk."""
        config = self.DATASET_CONFIGS.get(key)
        if not config:
            return None
        try:
            return load_from_disk(os.path.join(self.data_dir, config["folder_name"]))
        except Exception as e:
            logger.warning(f"Could not load {key}: {e}")
            return None

    def _standardize_subset(self, dataset: Dataset, count: int, text_col: str, label: int, context_name: str) -> \
            Optional[Dataset]:
        """Extracts subset, normalizes columns, and enforces schema with overflow checks."""
        total_len = len(dataset)
        if total_len == 0:
            return None

        # oversampling
        if count > total_len:
            sys.stderr.write(
                f"[WARNING] Requested {count} samples for '{context_name}', but only {total_len} available. Using all available.\n")
            take_n = total_len
        else:
            take_n = count

        subset = dataset.shuffle(seed=42).select(range(take_n))
        subset = subset.select_columns([text_col])

        if text_col != "text":
            subset = subset.rename_column(text_col, "text")

        subset = subset.add_column("label", [label] * len(subset))

        return subset.cast(Features({
            "text": Value("string"),
            "label": Value("int64")
        }))

    def generate_dataset(self, selection_counts: Optional[Dict[str, int]] = None, default_count: int = 20) -> Dataset:
        """
        Generates a unified dataset.

        Args:
            selection_counts: Dict {'dataset_name': count}. If None, uses all datasets with default_count.
            default_count: Number of samples to use if selection_counts is not provided.
        """
        chunks = []

        # if no specific selection is provided, iterate over all known configs
        if selection_counts is None:
            selection_counts = {k: default_count for k in self.DATASET_CONFIGS.keys()}

        for key, count in selection_counts.items():
            config = self.DATASET_CONFIGS.get(key)
            ds = self._load_dataset(key)

            if not config or not ds or count <= 0:
                continue

            if config["mode"] == "single_harmful":
                chunk = self._standardize_subset(ds, count, config["text_col_source"], 1, f"{key}_harmful")
                if chunk: chunks.append(chunk)

            elif config["mode"] == "mixed":
                harmful = ds.filter(lambda x: x[config["filter_col"]] == config["filter_val_harmful"])
                safe = ds.filter(lambda x: x[config["filter_col"]] == config["filter_val_safe"])

                chunk_h = self._standardize_subset(harmful, count, config["text_col_source"], 1, f"{key}_harmful")
                chunk_s = self._standardize_subset(safe, count, config["text_col_source"], 0, f"{key}_safe")

                if chunk_h: chunks.append(chunk_h)
                if chunk_s: chunks.append(chunk_s)

        if not chunks:
            raise ValueError("Resulting dataset is empty.")

        return concatenate_datasets(chunks).shuffle(seed=42)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loader = CorpusBuilder(data_dir="./datasets")

    ds_base = loader.generate_dataset({
        "attaq": 5,
        "gandalf": 5,
        "wildguard": 5
    })
    print(f"Generated: {len(ds_base)}")  # 5 + 5 + (5 + 5) = 20

    # overflow
    ds_explicit = loader.generate_dataset({
        "attaq": 1000000,
        "wildguard": 5
    })
    print(f"Generated: {len(ds_explicit)}")

    # default request
    ds_default = loader.generate_dataset(default_count=10)
    print(f"Generated: {len(ds_default)}")  # 40
