from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    """Configuration for the evaluation dataset."""

    name: str = "mteb/stsbenchmark-sts"
    config: str | None = None
    split: str = "test"
    query_col: str = "sentence1"
    passage_col: str = "sentence2"
    score_col: str | None = "score"
    score_scale: float = 5.0
    # Pre-loaded data (dict of column-name -> list). When set, skip HF download.
    data: dict[str, list] | None = field(default=None, repr=False)


DATASET_PRESETS: dict[str, DatasetConfig] = {
    "sts": DatasetConfig(
        name="mteb/stsbenchmark-sts",
        split="test",
        query_col="sentence1",
        passage_col="sentence2",
        score_col="score",
        score_scale=5.0,
    ),
    "natural-questions": DatasetConfig(
        name="sentence-transformers/natural-questions",
        split="train",
        query_col="query",
        passage_col="answer",
        score_col=None,
    ),
    "msmarco": DatasetConfig(
        name="sentence-transformers/msmarco-bm25",
        config="triplet",
        split="train",
        query_col="query",
        passage_col="positive",
        score_col=None,
    ),
    "squad": DatasetConfig(
        name="sentence-transformers/squad",
        split="train",
        query_col="question",
        passage_col="answer",
        score_col=None,
    ),
    "trivia-qa": DatasetConfig(
        name="sentence-transformers/trivia-qa",
        split="train",
        query_col="query",
        passage_col="answer",
        score_col=None,
    ),
    "gooaq": DatasetConfig(
        name="sentence-transformers/gooaq",
        split="train",
        query_col="question",
        passage_col="answer",
        score_col=None,
    ),
    "hotpotqa": DatasetConfig(
        name="sentence-transformers/hotpotqa",
        config="triplet",
        split="train",
        query_col="anchor",
        passage_col="positive",
        score_col=None,
    ),
}
