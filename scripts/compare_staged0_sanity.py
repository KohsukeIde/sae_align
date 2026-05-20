#!/usr/bin/env python
"""Compare Stage D0 dataset sanity summaries across backends."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping


def load_sanity(path: Path) -> Mapping[str, object]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "sanity" in obj:
        obj = obj["sanity"]
    return obj


def metric_rows(label: str, sanity: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(name: str, stats: object) -> None:
        if isinstance(stats, dict):
            rows.append(
                {
                    "label": label,
                    "metric": name,
                    "mean": stats.get("mean"),
                    "std": stats.get("std"),
                    "q05": stats.get("q05"),
                    "q50": stats.get("q50"),
                    "q95": stats.get("q95"),
                    "positive_rate": stats.get("positive_rate"),
                }
            )

    add("world_delta", sanity.get("world_delta"))
    add("event_magnitude", sanity.get("event_magnitude"))
    rows.append(
        {
            "label": label,
            "metric": "event_prevalence_any",
            "mean": sanity.get("event_prevalence_any"),
            "std": None,
            "q05": None,
            "q50": None,
            "q95": None,
            "positive_rate": sanity.get("event_prevalence_any"),
        }
    )
    detect = sanity.get("detect", {})
    if isinstance(detect, dict):
        for channel, stats in sorted(detect.items()):
            add(f"detect_{channel}", stats)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(path: Path, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    metrics = ["world_delta", "event_magnitude", "detect_rgb", "detect_range", "detect_local"]
    labels = sorted({str(r["label"]) for r in rows})
    means = {
        (str(r["label"]), str(r["metric"])): float(r["mean"])
        for r in rows
        if r.get("mean") not in (None, "")
    }
    x = range(len(metrics))
    width = 0.8 / max(1, len(labels))
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, label in enumerate(labels):
        vals = [means.get((label, m), 0.0) for m in metrics]
        ax.bar([j + i * width for j in x], vals, width=width, label=label)
    ax.set_xticks([j + width * (len(labels) - 1) / 2 for j in x])
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_ylabel("mean")
    ax.set_title("Stage D0 generator sanity comparison")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs=2, type=Path, required=True, help="two .sanity.json or .metadata.json files")
    parser.add_argument("--labels", nargs=2, default=["a", "b"])
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for label, path in zip(args.labels, args.input):
        rows.extend(metric_rows(label, load_sanity(path)))
    write_csv(args.out_csv, rows)
    if args.out_png is not None:
        maybe_plot(args.out_png, rows)
    print(f"Wrote {args.out_csv}")
    if args.out_png is not None:
        print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
