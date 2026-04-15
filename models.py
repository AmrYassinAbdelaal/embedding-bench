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
    # Prefixes prepended to inputs at encode time. Many retrieval-tuned
    # encoders (E5, BGE, Nomic, mxbai, Arctic, ...) expect distinct
    # query/passage instructions for best quality.
    query_prefix: str = ""
    passage_prefix: str = ""
    # Some models (e.g. Harrier) ship named prompts in the sentence-transformers
    # config and should be invoked via encode(..., prompt_name=...) rather than
    # by manually prepending a string. Only used by the sbert backend.
    query_prompt_name: str | None = None
    passage_prompt_name: str | None = None
    # Some HF repos ship custom modeling code and require opting in.
    trust_remote_code: bool = False


REGISTRY: dict[str, ModelConfig] = {
    # --- Official sentence-transformers models ---
    "mpnet": ModelConfig(
        name="all-mpnet-base-v2",
        model_id="sentence-transformers/all-mpnet-base-v2",
        is_baseline=True,
    ),
    "minilm-l6": ModelConfig(
        name="all-MiniLM-L6-v2",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
    ),
    "minilm-l12": ModelConfig(
        name="all-MiniLM-L12-v2",
        model_id="sentence-transformers/all-MiniLM-L12-v2",
    ),
    "distilroberta": ModelConfig(
        name="all-distilroberta-v1",
        model_id="sentence-transformers/all-distilroberta-v1",
    ),
    "multi-qa-mpnet": ModelConfig(
        name="multi-qa-mpnet-base-dot-v1",
        model_id="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    ),
    "multi-qa-distilbert": ModelConfig(
        name="multi-qa-distilbert-cos-v1",
        model_id="sentence-transformers/multi-qa-distilbert-cos-v1",
    ),
    "multi-qa-minilm": ModelConfig(
        name="multi-qa-MiniLM-L6-cos-v1",
        model_id="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    ),
    "paraphrase-mpnet": ModelConfig(
        name="paraphrase-mpnet-base-v2",
        model_id="sentence-transformers/paraphrase-mpnet-base-v2",
    ),
    "paraphrase-minilm-l6": ModelConfig(
        name="paraphrase-MiniLM-L6-v2",
        model_id="sentence-transformers/paraphrase-MiniLM-L6-v2",
    ),
    "paraphrase-minilm-l3": ModelConfig(
        name="paraphrase-MiniLM-L3-v2",
        model_id="sentence-transformers/paraphrase-MiniLM-L3-v2",
    ),
    "paraphrase-multilingual-mpnet": ModelConfig(
        name="paraphrase-multilingual-mpnet-base-v2",
        model_id="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    ),
    "paraphrase-multilingual-minilm": ModelConfig(
        name="paraphrase-multilingual-MiniLM-L12-v2",
        model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ),
    "distiluse-multilingual": ModelConfig(
        name="distiluse-base-multilingual-cased-v2",
        model_id="sentence-transformers/distiluse-base-multilingual-cased-v2",
    ),
    "labse": ModelConfig(
        name="LaBSE",
        model_id="sentence-transformers/LaBSE",
    ),
    # --- BGE (BAAI) ---
    # v1.5 English BGE models: only the query is instructed; passages are raw.
    "bge-small": ModelConfig(
        name="bge-small-en-v1.5",
        model_id="BAAI/bge-small-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "bge-base": ModelConfig(
        name="bge-base-en-v1.5",
        model_id="BAAI/bge-base-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "bge-large": ModelConfig(
        name="bge-large-en-v1.5",
        model_id="BAAI/bge-large-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    # bge-m3 does not use an instruction prefix.
    "bge-m3": ModelConfig(
        name="bge-m3",
        model_id="BAAI/bge-m3",
    ),
    # --- E5 (intfloat) ---
    # All E5 models expect "query: " / "passage: " prefixes.
    "e5-small-v2": ModelConfig(
        name="e5-small-v2",
        model_id="intfloat/e5-small-v2",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    "e5-base-v2": ModelConfig(
        name="e5-base-v2",
        model_id="intfloat/e5-base-v2",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    "e5-large-v2": ModelConfig(
        name="e5-large-v2",
        model_id="intfloat/e5-large-v2",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    "multilingual-e5-small": ModelConfig(
        name="multilingual-e5-small",
        model_id="intfloat/multilingual-e5-small",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    "multilingual-e5-base": ModelConfig(
        name="multilingual-e5-base",
        model_id="intfloat/multilingual-e5-base",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    "multilingual-e5-large": ModelConfig(
        name="multilingual-e5-large",
        model_id="intfloat/multilingual-e5-large",
        query_prefix="query: ",
        passage_prefix="passage: ",
    ),
    # --- GTE (Alibaba) ---
    # GTE v1 models do not use prefixes.
    "gte-small": ModelConfig(
        name="gte-small",
        model_id="thenlper/gte-small",
    ),
    "gte-base": ModelConfig(
        name="gte-base",
        model_id="thenlper/gte-base",
    ),
    "gte-large": ModelConfig(
        name="gte-large",
        model_id="thenlper/gte-large",
    ),
    # --- Nomic ---
    # Nomic expects task-specific prefixes.
    "nomic-v1": ModelConfig(
        name="nomic-embed-text-v1",
        model_id="nomic-ai/nomic-embed-text-v1",
        query_prefix="search_query: ",
        passage_prefix="search_document: ",
        trust_remote_code=True,
    ),
    "nomic-v1.5": ModelConfig(
        name="nomic-embed-text-v1.5",
        model_id="nomic-ai/nomic-embed-text-v1.5",
        query_prefix="search_query: ",
        passage_prefix="search_document: ",
        trust_remote_code=True,
    ),
    # --- Mixedbread ---
    # mxbai instructs the query only.
    "mxbai-large": ModelConfig(
        name="mxbai-embed-large-v1",
        model_id="mixedbread-ai/mxbai-embed-large-v1",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    # --- Jina ---
    # Jina v2 does not use prefixes.
    "jina-v2-small": ModelConfig(
        name="jina-embeddings-v2-small-en",
        model_id="jinaai/jina-embeddings-v2-small-en",
        trust_remote_code=True,
    ),
    "jina-v2-base": ModelConfig(
        name="jina-embeddings-v2-base-en",
        model_id="jinaai/jina-embeddings-v2-base-en",
        trust_remote_code=True,
    ),
    # --- Microsoft Harrier-OSS-v1 ---
    # Decoder-only multilingual embedding family. Queries use the
    # `web_search_query` named prompt (shipped in the model's ST config);
    # passages are encoded raw.
    "harrier-270m": ModelConfig(
        name="harrier-oss-v1-270m",
        model_id="microsoft/harrier-oss-v1-270m",
        query_prompt_name="web_search_query",
    ),
    "harrier-0.6b": ModelConfig(
        name="harrier-oss-v1-0.6b",
        model_id="microsoft/harrier-oss-v1-0.6b",
        query_prompt_name="web_search_query",
    ),
    "harrier-27b": ModelConfig(
        name="harrier-oss-v1-27b",
        model_id="microsoft/harrier-oss-v1-27b",
        query_prompt_name="web_search_query",
    ),
    # --- Snowflake Arctic ---
    # Arctic instructs the query only.
    "arctic-xs": ModelConfig(
        name="snowflake-arctic-embed-xs",
        model_id="Snowflake/snowflake-arctic-embed-xs",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "arctic-s": ModelConfig(
        name="snowflake-arctic-embed-s",
        model_id="Snowflake/snowflake-arctic-embed-s",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "arctic-m": ModelConfig(
        name="snowflake-arctic-embed-m",
        model_id="Snowflake/snowflake-arctic-embed-m",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "arctic-l": ModelConfig(
        name="snowflake-arctic-embed-l",
        model_id="Snowflake/snowflake-arctic-embed-l",
        query_prefix="Represent this sentence for searching relevant passages: ",
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
