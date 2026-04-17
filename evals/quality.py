from __future__ import annotations

import numpy as np
from datasets import load_dataset
from scipy.stats import spearmanr

from dataset_config import DatasetConfig


def _normalize(emb: np.ndarray) -> np.ndarray:
    """L2-normalize each row."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


ALL_RETRIEVAL_METRICS = [
    "mrr",
    "map@5", "map@10",
    "ndcg@5", "ndcg@10",
    "precision@1", "precision@5", "precision@10",
    "recall@1", "recall@5", "recall@10",
]

DEFAULT_RETRIEVAL_METRICS = ["mrr", "recall@1", "recall@5", "recall@10"]


def _retrieval_metrics(
    emb_q: np.ndarray,
    emb_p: np.ndarray,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Compute retrieval metrics assuming query i matches passage i."""
    if metrics is None:
        metrics = DEFAULT_RETRIEVAL_METRICS

    emb_q = _normalize(emb_q)
    emb_p = _normalize(emb_p)

    # Similarity matrix: (num_queries, num_passages)
    sims = emb_q @ emb_p.T

    n = sims.shape[0]
    sorted_indices = np.argsort(-sims, axis=1)
    ranks = np.array([int(np.where(sorted_indices[i] == i)[0][0]) for i in range(n)])

    results: dict[str, float] = {}

    for m in metrics:
        if m == "mrr":
            results["mrr"] = round(float(np.mean(1.0 / (ranks + 1))), 4)

        elif m.startswith("recall@"):
            k = int(m.split("@")[1])
            results[m] = round(float(np.mean(ranks < k)), 4)

        elif m.startswith("precision@"):
            k = int(m.split("@")[1])
            # Single relevant doc per query: precision@k = 1/k if hit, else 0
            results[m] = round(float(np.mean((ranks < k) / k)), 4)

        elif m.startswith("map@"):
            k = int(m.split("@")[1])
            # Single relevant doc: AP = 1/(rank+1) if rank < k, else 0
            ap = np.where(ranks < k, 1.0 / (ranks + 1), 0.0)
            results[m] = round(float(np.mean(ap)), 4)

        elif m.startswith("ndcg@"):
            k = int(m.split("@")[1])
            # Single relevant doc: DCG = 1/log2(rank+2) if rank < k, else 0
            # ideal DCG = 1/log2(2) = 1.0
            dcg = np.where(ranks < k, 1.0 / np.log2(ranks + 2), 0.0)
            results[m] = round(float(np.mean(dcg)), 4)

    return results


def evaluate_quality(
    model,
    ds_cfg: DatasetConfig | None = None,
    max_pairs: int | None = None,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate embedding quality on a dataset.

    Returns a dict with either {"spearman": float} for scored datasets
    or selected retrieval metrics for pair datasets.
    """
    if ds_cfg is None:
        ds_cfg = DatasetConfig()

    if ds_cfg.data is not None:
        data = ds_cfg.data
    else:
        dataset = load_dataset(ds_cfg.name, ds_cfg.config, split=ds_cfg.split)
        data = {col: list(dataset[col]) for col in dataset.column_names}
    queries = list(data[ds_cfg.query_col])
    passages = list(data[ds_cfg.passage_col])

    if max_pairs is not None and len(queries) > max_pairs:
        queries = queries[:max_pairs]
        passages = passages[:max_pairs]

    emb_q = model.encode(queries, is_query=True)
    emb_p = model.encode(passages, is_query=False)

    if ds_cfg.score_col is not None:
        # Scored mode: Spearman correlation
        scores = list(data[ds_cfg.score_col])
        if max_pairs is not None and len(scores) > max_pairs:
            scores = scores[:max_pairs]
        gold_scores = [s / ds_cfg.score_scale for s in scores]

        cos_sims = np.sum(emb_q * emb_p, axis=1) / (
            np.linalg.norm(emb_q, axis=1) * np.linalg.norm(emb_p, axis=1)
        )

        correlation, _ = spearmanr(cos_sims, gold_scores)
        return {"spearman": round(float(correlation), 4)}

    # Pair mode: retrieval metrics
    return _retrieval_metrics(emb_q, emb_p, metrics=metrics)
