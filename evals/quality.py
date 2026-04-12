from __future__ import annotations

import numpy as np
from datasets import load_dataset
from scipy.stats import spearmanr


def evaluate_quality(model) -> float:
    """Return Spearman correlation on the STS Benchmark test set."""
    dataset = load_dataset("mteb/stsbenchmark-sts", split="test")
    sentences1 = list(dataset["sentence1"])
    sentences2 = list(dataset["sentence2"])
    gold_scores = [s / 5.0 for s in dataset["score"]]

    emb1 = model.encode(sentences1)
    emb2 = model.encode(sentences2)

    # Row-wise cosine similarity
    cos_sims = np.sum(emb1 * emb2, axis=1) / (
        np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1)
    )

    correlation, _ = spearmanr(cos_sims, gold_scores)
    return correlation
