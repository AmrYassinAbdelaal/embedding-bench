from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate


def _format_metrics(metrics: dict[str, float]) -> str:
    """Format a single dataset's metrics into a compact string."""
    if "spearman" in metrics:
        return f"{metrics['spearman']:.4f}"
    if "mrr" in metrics:
        return f"MRR={metrics['mrr']:.4f} R@1={metrics['recall@1']:.4f}"
    return "—"


def _flatten_result(r: dict[str, Any]) -> dict[str, Any]:
    """Flatten a single result dict into a flat key-value dict for CSV."""
    flat: dict[str, Any] = {"model": r["name"]}

    for ds_key, metrics in r.get("quality", {}).items():
        for metric_name, value in metrics.items():
            flat[f"{ds_key}/{metric_name}"] = value

    speed = r.get("speed")
    if speed:
        flat["speed_sent_per_s"] = speed["sentences_per_second"]
        flat["median_time_s"] = speed["median_seconds"]

    memory = r.get("memory_mb")
    if memory is not None:
        flat["memory_mb"] = memory

    return flat


def export_csv(results: list[dict[str, Any]], path: str) -> None:
    """Export results to a CSV file."""
    rows = [_flatten_result(r) for r in results]
    fieldnames = list(rows[0].keys())
    # Ensure all fields are captured
    for row in rows[1:]:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to {path}")


def plot_charts(results: list[dict[str, Any]], output_dir: str) -> None:
    """Generate and save benchmark charts."""
    os.makedirs(output_dir, exist_ok=True)
    models = [r["name"] for r in results]

    # --- Quality charts (one per dataset) ---
    ds_keys: list[str] = []
    for r in results:
        quality = r.get("quality")
        if quality:
            ds_keys = list(quality.keys())
            break

    for ds_key in ds_keys:
        first_metrics = None
        for r in results:
            m = r.get("quality", {}).get(ds_key)
            if m:
                first_metrics = m
                break
        if not first_metrics:
            continue

        if "spearman" in first_metrics:
            # Single bar chart for spearman
            values = [r.get("quality", {}).get(ds_key, {}).get("spearman", 0) for r in results]
            fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 5))
            bars = ax.bar(models, values, color="#4C72B0")
            ax.set_ylabel("Spearman Correlation")
            ax.set_title(f"Quality — {ds_key}")
            ax.set_ylim(0, 1)
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=9)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f"quality_{ds_key}.png"), dpi=150)
            plt.close(fig)
        else:
            # Grouped bar chart for retrieval metrics
            metric_names = ["mrr", "recall@1", "recall@5", "recall@10"]
            x = np.arange(len(models))
            width = 0.18
            colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

            fig, ax = plt.subplots(figsize=(max(8, len(models) * 2), 5))
            for i, (metric, color) in enumerate(zip(metric_names, colors)):
                values = [r.get("quality", {}).get(ds_key, {}).get(metric, 0) for r in results]
                offset = (i - 1.5) * width
                bars = ax.bar(x + offset, values, width, label=metric, color=color)
                for bar, v in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)
            ax.set_ylabel("Score")
            ax.set_title(f"Retrieval Quality — {ds_key}")
            ax.set_ylim(0, 1.15)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=30, ha="right")
            ax.legend()
            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f"quality_{ds_key}.png"), dpi=150)
            plt.close(fig)

    # --- Speed chart ---
    speed_values = [r.get("speed", {}).get("sentences_per_second", 0) for r in results]
    if any(v > 0 for v in speed_values):
        fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 5))
        bars = ax.bar(models, speed_values, color="#55A868")
        ax.set_ylabel("Sentences / second")
        ax.set_title("Encoding Speed")
        for bar, v in zip(bars, speed_values):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(v), ha="center", va="bottom", fontsize=9)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "speed.png"), dpi=150)
        plt.close(fig)

    # --- Memory chart ---
    mem_values = [r.get("memory_mb", 0) for r in results]
    if any(v > 0 for v in mem_values):
        fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 5))
        bars = ax.bar(models, mem_values, color="#C44E52")
        ax.set_ylabel("Peak Memory (MB)")
        ax.set_title("Memory Usage")
        for bar, v in zip(bars, mem_values):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(v), ha="center", va="bottom", fontsize=9)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "memory.png"), dpi=150)
        plt.close(fig)

    print(f"Charts saved to {output_dir}/")


def print_report(
    results: list[dict[str, Any]],
    baseline_name: Optional[str] = None,
    csv_path: Optional[str] = None,
    chart_dir: Optional[str] = None,
) -> None:
    """Print a formatted comparison table and optionally export CSV/charts."""
    # Discover dataset columns from the first result that has quality data
    ds_keys: list[str] = []
    for r in results:
        quality = r.get("quality")
        if quality:
            ds_keys = list(quality.keys())
            break

    headers = ["Model"]
    for ds_key in ds_keys:
        headers.append(f"Quality ({ds_key})")
    headers.extend(["Speed (sent/s)", "Median Time (s)", "Memory (MB)"])

    rows: list[list[Any]] = []

    for r in results:
        name = r["name"]
        if r.get("is_baseline"):
            name += " [B]"

        quality = r.get("quality", {})
        speed = r.get("speed")
        memory = r.get("memory_mb")

        row: list[Any] = [name]
        for ds_key in ds_keys:
            metrics = quality.get(ds_key)
            row.append(_format_metrics(metrics) if metrics else "—")
        row.extend([
            f"{speed['sentences_per_second']}" if speed else "—",
            f"{speed['median_seconds']}" if speed else "—",
            f"{memory}" if memory is not None else "—",
        ])
        rows.append(row)

    print()
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    if baseline_name:
        print(f"\n[B] = baseline ({baseline_name})")
    print()

    if csv_path:
        export_csv(results, csv_path)

    if chart_dir:
        plot_charts(results, chart_dir)
