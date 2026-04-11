from __future__ import annotations

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


def evaluate_quality(model: SentenceTransformer) -> float:
    """Return Spearman correlation on the STS Benchmark test set."""
    dataset = load_dataset("mteb/stsbenchmark-sts", split="test")
    sentences1 = list(dataset["sentence1"])
    sentences2 = list(dataset["sentence2"])
    scores = [s / 5.0 for s in dataset["score"]]

    evaluator = EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
    results = evaluator(model)
    return results["spearman_cosine"]
