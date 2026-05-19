#!/usr/bin/env python
"""Aggregate Stage C0 result directories.

Usage:
  python scripts/summarize_stagec0_grid.py --root outputs/stagec0_v1

The script recursively finds reports/stagec0_summary.csv and
reports/stagec0_decision_summary.json under --root and writes compact aggregate
CSV/JSON files to --root.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def maybe_float(x: object) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--expected-summary-files", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    summary_paths = sorted(root.glob("**/reports/stagec0_summary.csv"))
    decision_paths = sorted(root.glob("**/reports/stagec0_decision_summary.json"))
    if int(args.expected_summary_files) > 0 and len(summary_paths) != int(args.expected_summary_files):
        raise RuntimeError(
            f"Expected {args.expected_summary_files} Stage C0 summaries under {root}, found {len(summary_paths)}."
        )
    rows: List[Dict[str, object]] = []
    for path in summary_paths:
        rel = str(path.parent.parent.relative_to(root)) if path.parent.parent != root else "."
        for row in read_csv(path):
            out: Dict[str, object] = {"run_dir": rel}
            out.update(row)
            rows.append(out)
    write_csv(root / "stagec0_grid_summary.csv", rows)

    decision_rows: List[Dict[str, object]] = []
    for path in decision_paths:
        rel = str(path.parent.parent.relative_to(root)) if path.parent.parent != root else "."
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        row = {"run_dir": rel}
        row.update(obj)
        decision_rows.append(row)
    write_csv(root / "stagec0_decision_grid.csv", decision_rows)

    # Compact method-level aggregate for the most important deltas.
    agg: List[Dict[str, object]] = []
    methods = sorted({str(r.get("method", "")) for r in rows if r.get("method")})
    metrics = [
        "event_f1_delta_vs_uniform",
        "event_f1_delta_minus_stageb_ref",
        "event_f1_delta_over_stageb_ref",
        "event_f1_ood_delta_vs_uniform",
        "event_f1_ood_delta_minus_stageb_ref",
        "event_f1_ood_delta_over_stageb_ref",
        "changed_cell_f1_delta_vs_uniform",
        "changed_cell_f1_delta_minus_stageb_ref",
        "changed_cell_f1_delta_over_stageb_ref",
        "reconstruction_mse_delta_vs_uniform",
        "reconstruction_mse_delta_minus_stageb_ref",
        "reconstruction_mse_delta_over_stageb_ref",
    ]
    for method in methods:
        subset = [r for r in rows if str(r.get("method")) == method]
        out: Dict[str, object] = {"method": method, "n_cells": len(subset)}
        for metric in metrics:
            vals = [maybe_float(r.get(metric)) for r in subset]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            if vals:
                arr = np.asarray(vals, dtype=np.float32)
                out[f"{metric}_mean"] = float(np.mean(arr))
                out[f"{metric}_std"] = float(np.std(arr))
                out[f"{metric}_min"] = float(np.min(arr))
                out[f"{metric}_max"] = float(np.max(arr))
                out[f"{metric}_positive_count"] = int(np.sum(arr > 0))
        agg.append(out)
    write_csv(root / "stagec0_method_delta_summary.csv", agg)

    # Compact per-cell decision table. This does not replace the preregistered
    # summary CSV; it only makes the important pairwise gates explicit.
    decision_cells: List[Dict[str, object]] = []
    by_run: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_run.setdefault(str(row.get("run_dir", ".")), []).append(row)
    for run_dir, run_rows in sorted(by_run.items()):
        by_method = {str(r.get("method")): r for r in run_rows}
        obs = by_method.get("observability")
        if obs is None:
            continue
        uniform = by_method.get("uniform", {})
        change = by_method.get("change_mask", {})
        shuffled = by_method.get("shuffled_observability", {})
        oracle = by_method.get("oracle_event", {})

        def f(row: Mapping[str, object], key: str) -> float:
            val = maybe_float(row.get(key))
            return float("nan") if val is None else float(val)

        obs_event = f(obs, "event_f1_mean")
        obs_ood = f(obs, "event_f1_ood_mean")
        best_nonoracle_event = np.nanmax(
            [f(uniform, "event_f1_mean"), f(change, "event_f1_mean"), f(shuffled, "event_f1_mean")]
        )
        best_nonoracle_ood = np.nanmax(
            [f(uniform, "event_f1_ood_mean"), f(change, "event_f1_ood_mean"), f(shuffled, "event_f1_ood_mean")]
        )
        obs_delta_event = f(obs, "event_f1_delta_vs_uniform")
        obs_delta_ood = f(obs, "event_f1_ood_delta_vs_uniform")
        row = {
            "run_dir": run_dir,
            "observability_event_f1": obs_event,
            "observability_event_f1_ood": obs_ood,
            "observability_event_delta_vs_uniform": obs_delta_event,
            "observability_ood_delta_vs_uniform": obs_delta_ood,
            "observability_minus_change_event": obs_event - f(change, "event_f1_mean"),
            "observability_minus_shuffled_event": obs_event - f(shuffled, "event_f1_mean"),
            "observability_minus_best_nonoracle_event": obs_event - best_nonoracle_event,
            "observability_minus_best_nonoracle_ood": obs_ood - best_nonoracle_ood,
            "observability_event_delta_minus_stageb_ref": f(obs, "event_f1_delta_minus_stageb_ref"),
            "observability_ood_delta_minus_stageb_ref": f(obs, "event_f1_ood_delta_minus_stageb_ref"),
            "oracle_event_minus_uniform_event": f(oracle, "event_f1_delta_vs_uniform"),
            "uniform_reconstruction_mse": f(uniform, "reconstruction_mse_mean"),
            "observability_reconstruction_mse_delta_vs_uniform": f(obs, "reconstruction_mse_delta_vs_uniform"),
            "go_event_or_ood_vs_uniform": bool(obs_delta_event > 0.0 or obs_delta_ood > 0.0),
            "go_event_or_ood_exceeds_stageb_ref": bool(obs_delta_event > 0.05 or obs_delta_ood > 0.05),
            "go_beats_change_event": bool(obs_event > f(change, "event_f1_mean")),
            "go_beats_shuffled_event": bool(obs_event > f(shuffled, "event_f1_mean")),
            "go_beats_best_nonoracle_event": bool(obs_event > best_nonoracle_event),
        }
        decision_cells.append(row)
    write_csv(root / "stagec0_pairwise_decision_summary.csv", decision_cells)

    decision = {
        "n_summary_files": len(summary_paths),
        "n_decision_files": len(decision_paths),
        "aggregate_files": [
            "stagec0_grid_summary.csv",
            "stagec0_decision_grid.csv",
            "stagec0_method_delta_summary.csv",
            "stagec0_pairwise_decision_summary.csv",
        ],
    }
    with open(root / "stagec0_grid_summary.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, sort_keys=True)
    print(f"Aggregated {len(summary_paths)} Stage C0 summaries under {root}")


if __name__ == "__main__":
    main()
