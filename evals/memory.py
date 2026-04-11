from __future__ import annotations

import multiprocessing
import os


def _measure(model_id: str, sentences: list[str], batch_size: int, queue: multiprocessing.Queue) -> None:
    import psutil
    from sentence_transformers import SentenceTransformer

    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss
    model = SentenceTransformer(model_id)
    model.encode(sentences, batch_size=batch_size, show_progress_bar=False)
    peak = process.memory_info().rss
    queue.put(peak - baseline)


def evaluate_memory(model_id: str, sentences: list[str], batch_size: int = 64) -> float:
    """Return memory delta in MB, measured in an isolated subprocess."""
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_measure, args=(model_id, sentences, batch_size, q))
    p.start()
    p.join()
    bytes_delta = q.get()
    return round(bytes_delta / (1024 * 1024), 1)
