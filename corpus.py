from __future__ import annotations

from datasets import load_dataset


def build_corpus(size: int) -> list[str]:
    """Build a corpus of real sentences from the STS Benchmark dataset."""
    dataset = load_dataset("mteb/stsbenchmark-sts", split="test")
    sentences = list(dataset["sentence1"]) + list(dataset["sentence2"])
    full: list[str] = []
    while len(full) < size:
        full.extend(sentences)
    return full[:size]
