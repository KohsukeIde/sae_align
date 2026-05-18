#!/usr/bin/env python
"""Aggregate Stage B.5 held-out alignment grid outputs."""

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
    seed = ""
    split = ""
    for part in rel.parts:
        if part.startswith("seed_"):
            seed = part.removeprefix("seed_")
        if part.startswith("split_"):
            split = part.removeprefix("split_")
    return {"data_seed": seed, "split_seed": split, "run": str(report_dir.parent.parent)}


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root
    report_dirs = sorted(root.glob("seed_*/split_*/heldout_alignment/reports"))
    gate_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    diff_rows: list[dict[str, object]] = []
    corr_rows: list[dict[str, object]] = []
    tie_rows: list[dict[str, object]] = []
    for report_dir in report_dirs:
        meta = run_metadata(report_dir, root)
        for name, target in [
            ("gate_summary.csv", gate_rows),
            ("heldout_same_action_cross_channel_alignment.csv", overlap_rows),
            ("heldout_action_effect_vs_static.csv", diff_rows),
            ("observability_score_correlation.csv", corr_rows),
            ("feature_tie_diagnostics.csv", tie_rows),
        ]:
            path = report_dir / name
            if path.exists():
                for row in read_csv(path):
                    target.append({**meta, **row})

    write_csv(out / "stageb5_gate_summary.csv", gate_rows)
    write_csv(out / "stageb5_cross_channel_alignment.csv", overlap_rows)
    write_csv(out / "stageb5_action_effect_vs_static.csv", diff_rows)
    write_csv(out / "stageb5_observability_score_correlation.csv", corr_rows)
    write_csv(out / "stageb5_feature_tie_diagnostics.csv", tie_rows)

    summary_rows: list[dict[str, object]] = []
    for rep in sorted({str(row["representation"]) for row in gate_rows}):
        for norm in sorted({str(row["normalization"]) for row in gate_rows if str(row["representation"]) == rep}):
            rows = [row for row in gate_rows if str(row["representation"]) == rep and str(row["normalization"]) == norm]
            if not rows:
                continue
            item: dict[str, object] = {
                "representation": rep,
                "normalization": norm,
                "n_runs": int(len(rows)),
                "normalization_primary": rows[0].get("normalization_primary", ""),
            }
            for field in [
                "gate0_core_redundancy_point_pass",
                "gate0_core_redundancy_ci_pass",
                "gate0_core_redundancy_gt_shuffled_ci_pass",
                "gate1_rgb_range_ci_pass",
                "gate1_action_effect_gt_static_ci_pass",
                "gate1_action_effect_gt_shuffled_ci_pass",
                "stagec_candidate",
            ]:
                item[f"{field}_count"] = sum(as_bool(row.get(field)) for row in rows)
            for field in ["gate1_rgb_range_adjusted", "gate1_rgb_range_ci_low", "gate2_detect_geom_rank_spearman"]:
                values = np.asarray([as_float(row.get(field)) for row in rows], dtype=np.float32)
                item[f"{field}_mean"] = float(np.nanmean(values)) if values.size else float("nan")
                item[f"{field}_min"] = float(np.nanmin(values)) if values.size else float("nan")
                item[f"{field}_max"] = float(np.nanmax(values)) if values.size else float("nan")
            branches = {}
            for row in rows:
                branch = str(row.get("gate1_b2_branch", ""))
                branches[branch] = branches.get(branch, 0) + 1
            item["b2_branch_counts_json"] = json.dumps(branches, sort_keys=True)
            summary_rows.append(item)
    write_csv(out / "stageb5_decision_summary.csv", summary_rows)

    tie_summary_rows: list[dict[str, object]] = []
    for rep in sorted({str(row["representation"]) for row in tie_rows}):
        for norm in sorted({str(row["normalization"]) for row in tie_rows if str(row["representation"]) == rep}):
            rows = [row for row in tie_rows if str(row["representation"]) == rep and str(row["normalization"]) == norm]
            if not rows:
                continue
            boundary = np.asarray([as_float(row.get("boundary_tie_fraction")) for row in rows], dtype=np.float32)
            duplicate = np.asarray([as_float(row.get("duplicate_fraction_rounded_1e8")) for row in rows], dtype=np.float32)
            tie_summary_rows.append(
                {
                    "representation": rep,
                    "normalization": norm,
                    "n_rows": int(len(rows)),
                    "boundary_tie_fraction_mean": float(np.nanmean(boundary)),
                    "boundary_tie_fraction_max": float(np.nanmax(boundary)),
                    "duplicate_fraction_mean": float(np.nanmean(duplicate)),
                    "duplicate_fraction_max": float(np.nanmax(duplicate)),
                }
            )
    write_csv(out / "stageb5_tie_summary.csv", tie_summary_rows)
    with open(out / "stageb5_grid_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "root": str(root),
                "n_runs": len(report_dirs),
                "summary_files": [
                    "stageb5_gate_summary.csv",
                    "stageb5_decision_summary.csv",
                    "stageb5_tie_summary.csv",
                    "stageb5_cross_channel_alignment.csv",
                    "stageb5_observability_score_correlation.csv",
                ],
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    print(f"Wrote Stage B.5 grid summary to {out}")


if __name__ == "__main__":
    main()
