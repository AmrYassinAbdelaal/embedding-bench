from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    model_id: str
    is_baseline: bool = False


REGISTRY: dict[str, ModelConfig] = {
    "mpnet": ModelConfig(
        name="all-mpnet-base-v2",
        model_id="sentence-transformers/all-mpnet-base-v2",
        is_baseline=True,
    ),
    "bge-small": ModelConfig(
        name="bge-small-en-v1.5",
        model_id="BAAI/bge-small-en-v1.5",
    ),
}
