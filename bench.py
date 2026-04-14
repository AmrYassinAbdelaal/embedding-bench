from __future__ import annotations

import argparse

from corpus import build_corpus
from dataset_config import DATASET_PRESETS, DatasetConfig
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

    # Dataset configuration
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sts"],
        choices=list(DATASET_PRESETS.keys()),
        help=f"Dataset presets to evaluate (default: sts). "
             f"Available: {', '.join(DATASET_PRESETS.keys())}",
    )
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Limit number of pairs per dataset (useful for large datasets)")

    # Custom dataset (overrides --datasets)
    parser.add_argument("--dataset", default=None,
                        help="Custom HF dataset name (overrides --datasets)")
    parser.add_argument("--config", default=None,
                        help="Dataset config/subset name (e.g. 'triplet')")
    parser.add_argument("--split", default="test")
    parser.add_argument("--query-col", default="sentence1")
    parser.add_argument("--passage-col", default="sentence2")
    parser.add_argument("--score-col", default="score",
                        help="Score column name. Pass 'none' for pair-only datasets.")
    parser.add_argument("--score-scale", type=float, default=5.0)

    # Output options
    parser.add_argument("--csv", default=None, metavar="PATH",
                        help="Export results to a CSV file")
    parser.add_argument("--charts", default=None, metavar="DIR",
                        help="Save charts to a directory (e.g. ./results)")

    args = parser.parse_args(argv)

    # Build list of dataset configs
    if args.dataset:
        # Custom dataset overrides presets
        ds_configs = [DatasetConfig(
            name=args.dataset,
            config=args.config,
            split=args.split,
            query_col=args.query_col,
            passage_col=args.passage_col,
            score_col=None if args.score_col.lower() == "none" else args.score_col,
            score_scale=args.score_scale,
        )]
    else:
        ds_configs = [DATASET_PRESETS[k] for k in args.datasets]

    configs = [REGISTRY[k] for k in args.models]
    baseline_name = next((c.name for c in configs if c.is_baseline), None)

    # Use first dataset for corpus building
    corpus: list[str] | None = None
    if not args.skip_speed or not args.skip_memory:
        print(f"Preparing corpus ({args.corpus_size} sentences)...")
        corpus = build_corpus(args.corpus_size, ds_configs[0])

    results = []
    for cfg in configs:
        print(f"\n{'='*50}")
        print(f"Benchmarking: {cfg.name}")
        print(f"{'='*50}")

        result: dict = {"name": cfg.name, "is_baseline": cfg.is_baseline}

        if not args.skip_quality:
            model = load_model(cfg)
            quality_results = {}
            for ds_cfg in ds_configs:
                ds_key = ds_cfg.name.split("/")[-1]
                print(f"  Evaluating quality on {ds_cfg.name}...")
                quality_results[ds_key] = evaluate_quality(
                    model, ds_cfg, max_pairs=args.max_pairs,
                )
                print(f"    {quality_results[ds_key]}")
            result["quality"] = quality_results
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

    print_report(results, baseline_name=baseline_name,
                 csv_path=args.csv, chart_dir=args.charts)


if __name__ == "__main__":
    main()
