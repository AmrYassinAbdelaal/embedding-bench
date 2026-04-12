from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    model_id: str
    is_baseline: bool = False
    backend: str = "sbert"
    gguf_file: str | None = None


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
    "bge-small-fe": ModelConfig(
        name="bge-small-en-v1.5 (fastembed)",
        model_id="BAAI/bge-small-en-v1.5",
        backend="fastembed",
    ),
    "all-minilm-fe": ModelConfig(
        name="all-MiniLM-L6-v2 (fastembed)",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        backend="fastembed",
    ),
}
