#!/usr/bin/env python
"""Aggregate Stage B.4 split-half reliability grid outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def run_metadata(report_dir: Path, root: Path) -> dict[str, object]:
    rel = report_dir.relative_to(root)
    parts = rel.parts
    seed = ""
    split = ""
    for part in parts:
        if part.startswith("seed_"):
            seed = part.removeprefix("seed_")
        if part.startswith("split_"):
            split = part.removeprefix("split_")
    return {"data_seed": seed, "split_seed": split, "run": str(report_dir.parent.parent)}


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root
    report_dirs = sorted(root.glob("seed_*/split_*/reliability/reports"))
    gate_rows: list[dict[str, object]] = []
    tie_rows: list[dict[str, object]] = []
    for report_dir in report_dirs:
        meta = run_metadata(report_dir, root)
        gate_path = report_dir / "gate_summary_ci.csv"
        if gate_path.exists():
            for row in read_csv(gate_path):
                gate_rows.append({**meta, **row})
        tie_path = report_dir / "feature_tie_diagnostics.csv"
        if tie_path.exists():
            for row in read_csv(tie_path):
                tie_rows.append({**meta, **row})

    write_csv(out / "stageb4_gate_summary_ci.csv", gate_rows)
    write_csv(out / "stageb4_feature_tie_diagnostics.csv", tie_rows)

    summary_rows: list[dict[str, object]] = []
    norms = sorted({str(row["normalization"]) for row in gate_rows})
    for norm in norms:
        rows = [row for row in gate_rows if str(row["normalization"]) == norm]
        if not rows:
            continue
        fields = [
            "gate_minus1_identity_ci_pass",
            "gate_minus1_same_channel_core_ci_pass",
            "gate_minus1_same_channel_all_ci_pass",
            "gate_minus1_same_vs_shuffled_ci_pass",
            "gate0_redundancy_core_ci_pass",
            "gate0_redundancy_all_ci_pass",
        ]
        item: dict[str, object] = {
            "normalization": norm,
            "n_runs": len(rows),
            "normalization_primary": rows[0].get("normalization_primary", ""),
            "normalization_transductive": rows[0].get("normalization_transductive", ""),
        }
        for field in fields:
            item[f"{field}_count"] = sum(as_bool(row.get(field)) for row in rows)
        for field in [
            "gate_minus1_same_channel_core_ci_low_min",
            "gate_minus1_same_channel_all_ci_low_min",
            "gate_minus1_same_vs_shuffled_diff_ci_low_min",
            "gate0_redundancy_core_ci_low_min",
            "gate0_redundancy_all_ci_low_min",
        ]:
            values = np.asarray([as_float(row.get(field)) for row in rows], dtype=np.float32)
            item[f"{field}_mean"] = float(np.nanmean(values)) if values.size else float("nan")
            item[f"{field}_min"] = float(np.nanmin(values)) if values.size else float("nan")
        summary_rows.append(item)
    write_csv(out / "stageb4_decision_summary.csv", summary_rows)

    tie_summary_rows: list[dict[str, object]] = []
    for norm in sorted({str(row["normalization"]) for row in tie_rows}):
        for feature_set in sorted({str(row["feature_set"]) for row in tie_rows}):
            rows = [row for row in tie_rows if str(row["normalization"]) == norm and str(row["feature_set"]) == feature_set]
            if not rows:
                continue
            boundary = np.asarray([as_float(row.get("boundary_tie_fraction")) for row in rows], dtype=np.float32)
            duplicate = np.asarray([as_float(row.get("duplicate_fraction_rounded_1e8")) for row in rows], dtype=np.float32)
            tie_summary_rows.append(
                {
                    "normalization": norm,
                    "feature_set": feature_set,
                    "n_rows": int(len(rows)),
                    "boundary_tie_fraction_mean": float(np.nanmean(boundary)),
                    "boundary_tie_fraction_max": float(np.nanmax(boundary)),
                    "duplicate_fraction_mean": float(np.nanmean(duplicate)),
                    "duplicate_fraction_max": float(np.nanmax(duplicate)),
                }
            )
    write_csv(out / "stageb4_tie_summary.csv", tie_summary_rows)
    with open(out / "stageb4_grid_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "root": str(root),
                "n_runs": len(report_dirs),
                "normalizations": norms,
                "summary_files": [
                    "stageb4_gate_summary_ci.csv",
                    "stageb4_decision_summary.csv",
                    "stageb4_tie_summary.csv",
                    "stageb4_feature_tie_diagnostics.csv",
                ],
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    print(f"Wrote Stage B.4 grid summary to {out}")


if __name__ == "__main__":
    main()
