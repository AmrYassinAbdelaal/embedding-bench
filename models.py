from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


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
    # "bge-small-fe": ModelConfig(
    #     name="bge-small-en-v1.5 (fastembed)",
    #     model_id="BAAI/bge-small-en-v1.5",
    #     backend="fastembed",
    # ),
    # "all-minilm-fe": ModelConfig(
    #     name="all-MiniLM-L6-v2 (fastembed)",
    #     model_id="sentence-transformers/all-MiniLM-L6-v2",
    #     backend="fastembed",
    # ),
    # "bge-small-le": ModelConfig(
    #     name="bge-small-en-v1.5 (libembedding)",
    #     model_id="BAAI/bge-small-en-v1.5",
    #     backend="libembedding",
    # ),
    # "all-minilm-le": ModelConfig(
    #     name="all-MiniLM-L6-v2 (libembedding)",
    #     model_id="sentence-transformers/all-MiniLM-L6-v2",
    #     backend="libembedding",
    # ),
}

VALID_BACKENDS = {"sbert", "fastembed", "libembedding", "gguf"}
CUSTOM_MODELS_PATH = Path(__file__).parent / "custom_models.json"


def register_model(key: str, config: ModelConfig) -> None:
    if key in REGISTRY:
        raise ValueError(f"Model key '{key}' already exists in registry")
    if config.backend not in VALID_BACKENDS:
        raise ValueError(f"Invalid backend '{config.backend}'. Must be one of: {VALID_BACKENDS}")
    REGISTRY[key] = config


def load_custom_models_from_file(path: Path = CUSTOM_MODELS_PATH) -> None:
    if not path.exists():
        return
    with open(path) as f:
        entries = json.load(f)
    for key, fields in entries.items():
        if key not in REGISTRY:
            REGISTRY[key] = ModelConfig(**fields)


def save_custom_model_to_file(key: str, config: ModelConfig, path: Path = CUSTOM_MODELS_PATH) -> None:
    existing: dict = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing[key] = asdict(config)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
