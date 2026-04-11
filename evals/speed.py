from __future__ import annotations

import statistics
import time

from sentence_transformers import SentenceTransformer


def evaluate_speed(
    model: SentenceTransformer,
    sentences: list[str],
    num_runs: int = 3,
    batch_size: int = 64,
) -> dict[str, float]:
    """Measure encoding latency. Returns median time and throughput."""
    model.encode(sentences, batch_size=batch_size, show_progress_bar=False)

    times: list[float] = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model.encode(sentences, batch_size=batch_size, show_progress_bar=False)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_time = statistics.median(times)
    return {
        "median_seconds": round(median_time, 4),
        "sentences_per_second": round(len(sentences) / median_time, 1),
    }
