#!/usr/bin/env python
"""Aggregate Stage B.6 diagnostic grid outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PRIMARY_REPRESENTATION = "pca_probe_only"
PRIMARY_NORMALIZATION = "probe_action_type_apply"
PRIMARY_COMPONENTS = 32
PRIMARY_K = 10
PRIMARY_JITTER = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--expected-report-dirs", type=int, default=None)
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


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def run_metadata(report_dir: Path, root: Path) -> dict[str, object]:
    rel = report_dir.relative_to(root)
    seed = ""
    split = ""
    pca = ""
    for part in rel.parts:
        if part.startswith("seed_"):
            seed = part.removeprefix("seed_")
        if part.startswith("split_"):
            split = part.removeprefix("split_")
        if part.startswith("pca_"):
            pca = part.removeprefix("pca_")
    return {
        "data_seed": seed,
        "split_seed": split,
        "pca_dim": pca,
        "run": str(report_dir.parent.parent),
    }


def finite_values(rows: list[dict[str, object]], field: str) -> np.ndarray:
    values = np.asarray([as_float(row.get(field)) for row in rows], dtype=np.float64)
    return values[np.isfinite(values)]


def mean_min_max(rows: list[dict[str, object]], field: str) -> dict[str, float]:
    values = finite_values(rows, field)
    if values.size == 0:
        return {f"{field}_mean": float("nan"), f"{field}_min": float("nan"), f"{field}_max": float("nan")}
    return {
        f"{field}_mean": float(values.mean()),
        f"{field}_min": float(values.min()),
        f"{field}_max": float(values.max()),
    }


def group_key(row: dict[str, object], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in fields)


def aggregate_decisions(knn_rows: list[dict[str, object]], diff_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    target = [
        row
        for row in knn_rows
        if row.get("control") == "action_effect_heldout_signature"
        and row.get("channel_a") == "rgb"
        and row.get("channel_b") == "range"
    ]
    diff_target = [
        row
        for row in diff_rows
        if row.get("channel_a") == "rgb" and row.get("channel_b") == "range"
    ]
    fields = ("representation", "representation_components", "normalization")
    out = []
    for key in sorted({group_key(row, fields) for row in target}):
        rows = [row for row in target if group_key(row, fields) == key]
        if not rows:
            continue
        static_diffs = [
            row
            for row in diff_target
            if group_key(row, fields) == key and row.get("comparison") == "action_effect_minus_static"
        ]
        shuffled_diffs = [
            row
            for row in diff_target
            if group_key(row, fields) == key and row.get("comparison") == "action_effect_minus_shuffled"
        ]
        item: dict[str, object] = {
            "representation": key[0],
            "representation_components": key[1],
            "normalization": key[2],
            "n_rows": int(len(rows)),
            "primary_cell": bool(
                key[0] == PRIMARY_REPRESENTATION
                and int(float(key[1])) == PRIMARY_COMPONENTS
                and key[2] == PRIMARY_NORMALIZATION
            ),
        }
        item.update(mean_min_max(rows, "chance_adjusted_overlap"))
        item["positive_fraction"] = float(
            np.mean([as_float(row.get("chance_adjusted_overlap")) > 0.0 for row in rows])
        )
        item["k_values_json"] = json.dumps(sorted({int(float(row.get("k", 0))) for row in rows}))
        item["jitter_epsilons_json"] = json.dumps(sorted({as_float(row.get("jitter_epsilon")) for row in rows}))
        for name, subset in [
            ("ae_minus_static", static_diffs),
            ("ae_minus_shuffled", shuffled_diffs),
        ]:
            item.update(mean_min_max(subset, "diff_mean"))
            item[f"{name}_positive_fraction"] = float(
                np.mean([as_float(row.get("diff_mean")) > 0.0 for row in subset])
            ) if subset else float("nan")
            item[f"{name}_ci_low_positive_fraction"] = float(
                np.mean([as_float(row.get("diff_ci95_low")) > 0.0 for row in subset])
            ) if subset else float("nan")
        out.append(item)
    return out


def aggregate_primary_cell(
    knn_rows: list[dict[str, object]],
    diff_rows: list[dict[str, object]],
    corr_rows: list[dict[str, object]],
    sanity_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    def is_primary(row: dict[str, object]) -> bool:
        return (
            row.get("representation") == PRIMARY_REPRESENTATION
            and int(float(row.get("representation_components", -1))) == PRIMARY_COMPONENTS
            and row.get("normalization") == PRIMARY_NORMALIZATION
            and int(float(row.get("k", PRIMARY_K))) == PRIMARY_K
            and abs(as_float(row.get("jitter_epsilon", PRIMARY_JITTER)) - PRIMARY_JITTER) < 1e-12
        )

    rgb_range = [
        row
        for row in knn_rows
        if is_primary(row)
        and row.get("control") == "action_effect_heldout_signature"
        and row.get("channel_a") == "rgb"
        and row.get("channel_b") == "range"
    ]
    red = [
        row
        for row in knn_rows
        if is_primary(row)
        and row.get("control") == "action_effect_heldout_signature"
        and (row.get("channel_a"), row.get("channel_b")) in {("rgb", "noisy_rgb"), ("rgb", "gray_rgb"), ("rgb", "blur_rgb")}
    ]
    diffs = [
        row
        for row in diff_rows
        if is_primary(row) and row.get("channel_a") == "rgb" and row.get("channel_b") == "range"
    ]
    obs = [
        row
        for row in corr_rows
        if is_primary(row)
        and row.get("channel_a") == "rgb"
        and row.get("channel_b") == "range"
        and row.get("score") == "detect_geom_rank_mean"
    ]
    sanity = [
        row
        for row in sanity_rows
        if row.get("representation") == PRIMARY_REPRESENTATION
        and int(float(row.get("representation_components", -1))) == PRIMARY_COMPONENTS
        and row.get("normalization") == PRIMARY_NORMALIZATION
        and row.get("channel_a") == "rgb"
        and row.get("channel_b") == "range"
        and row.get("control") == "action_effect_heldout_signature"
    ]
    sanity_nonridge = [
        row
        for row in sanity
        if "ridge" not in str(row.get("measurement", "")).lower()
    ]
    sanity_ridge = [
        row
        for row in sanity
        if "ridge" in str(row.get("measurement", "")).lower()
    ]
    rgb_values = finite_values(rgb_range, "chance_adjusted_overlap")
    obs_values = finite_values(obs, "spearman")
    sanity_values = finite_values(sanity_nonridge, "calibrated_score")
    sanity_all_values = finite_values(sanity, "calibrated_score")
    sanity_ridge_values = finite_values(sanity_ridge, "calibrated_score")
    return [
        {
            "primary_replication_cell": f"{PRIMARY_REPRESENTATION}/{PRIMARY_NORMALIZATION}/d{PRIMARY_COMPONENTS}/k{PRIMARY_K}/jitter0",
            "n_runs": int(len(rgb_range)),
            "rgb_range_adjusted_mean": float(rgb_values.mean()) if rgb_values.size else float("nan"),
            "rgb_range_adjusted_min": float(rgb_values.min()) if rgb_values.size else float("nan"),
            "rgb_range_positive_count": int(sum(as_float(row.get("chance_adjusted_overlap")) > 0.0 for row in rgb_range)),
            "redundancy_positive_count": int(sum(as_float(row.get("chance_adjusted_overlap")) > 0.0 for row in red)),
            "ae_gt_static_ci_count": int(
                sum(
                    row.get("comparison") == "action_effect_minus_static"
                    and as_float(row.get("diff_ci95_low")) > 0.0
                    for row in diffs
                )
            ),
            "ae_gt_shuffled_ci_count": int(
                sum(
                    row.get("comparison") == "action_effect_minus_shuffled"
                    and as_float(row.get("diff_ci95_low")) > 0.0
                    for row in diffs
                )
            ),
            "detect_geom_spearman_mean": float(obs_values.mean()) if obs_values.size else float("nan"),
            "measurement_sanity_calibrated_mean": float(sanity_values.mean()) if sanity_values.size else float("nan"),
            "measurement_sanity_all_calibrated_mean": float(sanity_all_values.mean())
            if sanity_all_values.size
            else float("nan"),
            "ridge_sanity_calibrated_mean": float(sanity_ridge_values.mean())
            if sanity_ridge_values.size
            else float("nan"),
        }
    ]


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root
    report_dirs = sorted(root.glob("seed_*/split_*/pca_*/stageb6_diagnostics/reports"))
    if args.expected_report_dirs is not None and len(report_dirs) != int(args.expected_report_dirs):
        raise RuntimeError(
            f"Expected {int(args.expected_report_dirs)} Stage B.6 report dirs under {root}, "
            f"found {len(report_dirs)}. Refusing to aggregate an incomplete grid."
        )
    knn_rows: list[dict[str, object]] = []
    diff_rows: list[dict[str, object]] = []
    corr_rows: list[dict[str, object]] = []
    tie_rows: list[dict[str, object]] = []
    sanity_rows: list[dict[str, object]] = []
    literature_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    for report_dir in report_dirs:
        meta = run_metadata(report_dir, root)
        for name, target in [
            ("b6_knn_sensitivity.csv", knn_rows),
            ("b6_paired_differences.csv", diff_rows),
            ("b6_observability_score_correlation_by_k_jitter.csv", corr_rows),
            ("b6_feature_tie_by_k_jitter.csv", tie_rows),
            ("b6_measurement_sanity.csv", sanity_rows),
            ("b6_literature_metrics.csv", literature_rows),
            ("b6_decision_summary.csv", decision_rows),
        ]:
            path = report_dir / name
            if path.exists():
                for row in read_csv(path):
                    target.append({**meta, **row})

    write_csv(out / "stageb6_knn_sensitivity.csv", knn_rows)
    write_csv(out / "stageb6_paired_differences.csv", diff_rows)
    write_csv(out / "stageb6_observability_score_correlation.csv", corr_rows)
    write_csv(out / "stageb6_feature_tie_by_k_jitter.csv", tie_rows)
    write_csv(out / "stageb6_measurement_sanity.csv", sanity_rows)
    write_csv(out / "stageb6_literature_metrics.csv", literature_rows)
    write_csv(out / "stageb6_run_decision_summary.csv", decision_rows)

    aggregate_rows = aggregate_decisions(knn_rows, diff_rows)
    primary_rows = aggregate_primary_cell(knn_rows, diff_rows, corr_rows, sanity_rows)
    write_csv(out / "stageb6_decision_summary.csv", aggregate_rows)
    write_csv(out / "stageb6_primary_cell_summary.csv", primary_rows)
    summary = {
        "root": str(root),
        "n_report_dirs": int(len(report_dirs)),
        "n_knn_rows": int(len(knn_rows)),
        "n_diff_rows": int(len(diff_rows)),
        "n_corr_rows": int(len(corr_rows)),
        "n_sanity_rows": int(len(sanity_rows)),
        "n_literature_metric_rows": int(len(literature_rows)),
        "primary_cell": primary_rows[0] if primary_rows else {},
        "notes": [
            "B.6 is diagnostic. Do not pick the best cell as primary evidence.",
            "pca_all_action rows are transductive diagnostic upper bounds.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "stageb6_grid_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"Wrote Stage B.6 grid summary to {out}")


if __name__ == "__main__":
    main()
