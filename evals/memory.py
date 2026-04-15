from __future__ import annotations

import json
import os
import subprocess
import sys


def _measure(model_id: str, backend: str, sentences: list[str], batch_size: int) -> float:
    import psutil
    from models import ModelConfig
    from wrapper import load_model

    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss
    cfg = ModelConfig(name="", model_id=model_id, backend=backend)
    model = load_model(cfg)
    model.encode(sentences, batch_size=batch_size)
    peak = process.memory_info().rss
    return round((peak - baseline) / (1024 * 1024), 1)


def evaluate_memory(model_id: str, sentences: list[str], batch_size: int = 64, backend: str = "sbert") -> float:
    """Return memory delta in MB, measured in an isolated subprocess."""
    payload = json.dumps({
        "model_id": model_id,
        "backend": backend,
        "sentences": sentences,
        "batch_size": batch_size,
    })
    result = subprocess.run(
        [sys.executable, "-m", "evals.memory"],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    args = json.loads(sys.stdin.read())
    mb = _measure(
        model_id=args["model_id"],
        backend=args["backend"],
        sentences=args["sentences"],
        batch_size=args["batch_size"],
    )
    print(mb)
