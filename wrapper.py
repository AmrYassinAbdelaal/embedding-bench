from __future__ import annotations

import numpy as np

from models import ModelConfig


class SBertWrapper:
    """Wraps sentence_transformers.SentenceTransformer."""

    def __init__(self, cfg: ModelConfig):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(cfg.model_id)

    def encode(self, sentences: list[str], batch_size: int = 64, **kwargs) -> np.ndarray:
        kwargs.setdefault("show_progress_bar", False)
        return self._model.encode(sentences, batch_size=batch_size, **kwargs)


class GGUFWrapper:
    """Wraps llama_cpp.Llama in embedding mode."""

    def __init__(self, cfg: ModelConfig):
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        path = hf_hub_download(repo_id=cfg.model_id, filename=cfg.gguf_file)
        self._model = Llama(
            model_path=path, embedding=True, n_ctx=512, verbose=False
        )

    def encode(self, sentences: list[str], batch_size: int = 64, **kwargs) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            response = self._model.create_embedding(batch)
            embeddings = [item["embedding"] for item in response["data"]]
            all_embeddings.extend(embeddings)
        return np.array(all_embeddings, dtype=np.float32)


class FastEmbedWrapper:
    """Wraps fastembed.TextEmbedding."""

    def __init__(self, cfg: ModelConfig):
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name=cfg.model_id)

    def encode(self, sentences: list[str], batch_size: int = 64, **kwargs) -> np.ndarray:
        embeddings = list(self._model.embed(sentences, batch_size=batch_size))
        return np.array(embeddings, dtype=np.float32)


class LibEmbedWrapper:
    """Wraps libembedding.TextEmbedding."""

    def __init__(self, cfg: ModelConfig):
        from libembedding import TextEmbedding
        self._model = TextEmbedding(cfg.model_id)

    def encode(self, sentences: list[str], batch_size: int = 64, **kwargs) -> np.ndarray:
        embeddings = list(self._model.embed(sentences, batch_size=batch_size))
        return np.array(embeddings, dtype=np.float32)


def load_model(cfg: ModelConfig) -> SBertWrapper | GGUFWrapper | FastEmbedWrapper | LibEmbedWrapper:
    """Factory: returns the right wrapper for the model's backend."""
    if cfg.backend == "gguf":
        return GGUFWrapper(cfg)
    if cfg.backend == "fastembed":
        return FastEmbedWrapper(cfg)
    if cfg.backend == "libembedding":
        return LibEmbedWrapper(cfg)
    return SBertWrapper(cfg)
