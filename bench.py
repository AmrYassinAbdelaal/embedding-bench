from __future__ import annotations

import argparse

from corpus import build_corpus
from evals import evaluate_memory, evaluate_quality, evaluate_speed
from models import REGISTRY
from report import print_report
from wrapper import load_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="embedding-bench",
        description="Compare embedding models on quality, speed, and memory.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(REGISTRY.keys()),
        choices=list(REGISTRY.keys()),
        help="Models to benchmark (default: all)",
    )
    parser.add_argument("--corpus-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")

    args = parser.parse_args(argv)

    configs = [REGISTRY[k] for k in args.models]
    baseline_name = next((c.name for c in configs if c.is_baseline), None)

    corpus: list[str] | None = None
    if not args.skip_speed or not args.skip_memory:
        print(f"Preparing corpus ({args.corpus_size} sentences)...")
        corpus = build_corpus(args.corpus_size)

    results = []
    for cfg in configs:
        print(f"\n{'='*50}")
        print(f"Benchmarking: {cfg.name}")
        print(f"{'='*50}")

        result: dict = {"name": cfg.name, "is_baseline": cfg.is_baseline}

        if not args.skip_quality:
            print("  Evaluating quality (STS Benchmark)...")
            model = load_model(cfg)
            result["quality"] = evaluate_quality(model)
            print(f"  Quality: {result['quality']:.4f}")
            del model

        if not args.skip_speed and corpus is not None:
            print(f"  Evaluating speed ({args.num_runs} runs, {args.corpus_size} sentences)...")
            model = load_model(cfg)
            result["speed"] = evaluate_speed(model, corpus, num_runs=args.num_runs, batch_size=args.batch_size)
            print(f"  Speed: {result['speed']['sentences_per_second']} sent/s")
            del model

        if not args.skip_memory and corpus is not None:
            print("  Evaluating memory (isolated subprocess)...")
            result["memory_mb"] = evaluate_memory(cfg.model_id, corpus, batch_size=args.batch_size, backend=cfg.backend)
            print(f"  Memory: {result['memory_mb']} MB")

        results.append(result)

    print_report(results, baseline_name=baseline_name)


if __name__ == "__main__":
    main()
