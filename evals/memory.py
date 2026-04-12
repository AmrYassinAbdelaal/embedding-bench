from __future__ import annotations

import multiprocessing
import os


def _measure(model_id: str, backend: str, sentences: list[str], batch_size: int, queue: multiprocessing.Queue) -> None:
    import psutil
    from models import ModelConfig
    from wrapper import load_model

    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss
    cfg = ModelConfig(name="", model_id=model_id, backend=backend)
    model = load_model(cfg)
    model.encode(sentences, batch_size=batch_size)
    peak = process.memory_info().rss
    queue.put(peak - baseline)


def evaluate_memory(model_id: str, sentences: list[str], batch_size: int = 64, backend: str = "sbert") -> float:
    """Return memory delta in MB, measured in an isolated subprocess."""
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_measure, args=(model_id, backend, sentences, batch_size, q))
    p.start()
    p.join()
    bytes_delta = q.get()
    return round(bytes_delta / (1024 * 1024), 1)
