from __future__ import annotations

from datasets import load_dataset

from dataset_config import DatasetConfig


def build_corpus(size: int, ds_cfg: DatasetConfig | None = None) -> list[str]:
    """Build a corpus of real sentences from the configured dataset."""
    if ds_cfg is None:
        ds_cfg = DatasetConfig()
    dataset = load_dataset(ds_cfg.name, ds_cfg.config, split=ds_cfg.split)
    sentences = list(dataset[ds_cfg.query_col]) + list(dataset[ds_cfg.passage_col])
    full: list[str] = []
    while len(full) < size:
        full.extend(sentences)
    return full[:size]
