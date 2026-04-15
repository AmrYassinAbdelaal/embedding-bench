from __future__ import annotations

import numpy as np
from datasets import load_dataset
from scipy.stats import spearmanr

from dataset_config import DatasetConfig


def _normalize(emb: np.ndarray) -> np.ndarray:
    """L2-normalize each row."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


def _retrieval_metrics(emb_q: np.ndarray, emb_p: np.ndarray) -> dict[str, float]:
    """Compute MRR and Recall@k assuming query i matches passage i."""
    emb_q = _normalize(emb_q)
    emb_p = _normalize(emb_p)

    # Similarity matrix: (num_queries, num_passages)
    sims = emb_q @ emb_p.T

    n = sims.shape[0]
    # For each query, rank passages by descending similarity
    # ranks[i] = rank of the correct passage (0-indexed)
    sorted_indices = np.argsort(-sims, axis=1)
    ranks = np.array([int(np.where(sorted_indices[i] == i)[0][0]) for i in range(n)])

    mrr = float(np.mean(1.0 / (ranks + 1)))
    recall_1 = float(np.mean(ranks < 1))
    recall_5 = float(np.mean(ranks < 5))
    recall_10 = float(np.mean(ranks < 10))

    return {
        "mrr": round(mrr, 4),
        "recall@1": round(recall_1, 4),
        "recall@5": round(recall_5, 4),
        "recall@10": round(recall_10, 4),
    }


def evaluate_quality(
    model,
    ds_cfg: DatasetConfig | None = None,
    max_pairs: int | None = None,
) -> dict[str, float]:
    """Evaluate embedding quality on a dataset.

    Returns a dict with either {"spearman": float} for scored datasets
    or {"mrr", "recall@1", "recall@5", "recall@10"} for pair datasets.
    """
    if ds_cfg is None:
        ds_cfg = DatasetConfig()

    dataset = load_dataset(ds_cfg.name, ds_cfg.config, split=ds_cfg.split)
    queries = list(dataset[ds_cfg.query_col])
    passages = list(dataset[ds_cfg.passage_col])

    if max_pairs is not None and len(queries) > max_pairs:
        queries = queries[:max_pairs]
        passages = passages[:max_pairs]

    emb_q = model.encode(queries, is_query=True)
    emb_p = model.encode(passages, is_query=False)

    if ds_cfg.score_col is not None:
        # Scored mode: Spearman correlation
        scores = list(dataset[ds_cfg.score_col])
        if max_pairs is not None and len(scores) > max_pairs:
            scores = scores[:max_pairs]
        gold_scores = [s / ds_cfg.score_scale for s in scores]

        cos_sims = np.sum(emb_q * emb_p, axis=1) / (
            np.linalg.norm(emb_q, axis=1) * np.linalg.norm(emb_p, axis=1)
        )

        correlation, _ = spearmanr(cos_sims, gold_scores)
        return {"spearman": round(float(correlation), 4)}

    # Pair mode: retrieval metrics
    return _retrieval_metrics(emb_q, emb_p)
