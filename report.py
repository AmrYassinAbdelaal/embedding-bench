from __future__ import annotations

from typing import Any, Optional

from tabulate import tabulate


def print_report(results: list[dict[str, Any]], baseline_name: Optional[str] = None) -> None:
    """Print a formatted comparison table to stdout."""
    headers = ["Model", "Quality (STS)", "Speed (sent/s)", "Median Time (s)", "Memory (MB)"]
    rows: list[list[Any]] = []

    for r in results:
        name = r["name"]
        if r.get("is_baseline"):
            name += " [B]"

        quality = r.get("quality")
        speed = r.get("speed")
        memory = r.get("memory_mb")

        rows.append([
            name,
            f"{quality:.4f}" if quality is not None else "—",
            f"{speed['sentences_per_second']}" if speed else "—",
            f"{speed['median_seconds']}" if speed else "—",
            f"{memory}" if memory is not None else "—",
        ])

    print()
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    if baseline_name:
        print(f"\n[B] = baseline ({baseline_name})")
    print()
