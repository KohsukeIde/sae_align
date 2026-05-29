#!/usr/bin/env python
"""Summarize static-weak channel-pair sweep outputs."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", "nan", "None"} else float("nan")


def pair_key(row: dict[str, str]) -> str:
    return f"{row['channel_a']}:{row['channel_b']}"


def avg(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(mean(vals)) if vals else float("nan")


def min_finite(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(min(vals)) if vals else float("nan")


def count_pos(values: Iterable[float]) -> int:
    return int(sum(float(v) > 0 for v in values if np.isfinite(float(v))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--static-control-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    static_dir = Path(args.static_control_dir) if args.static_control_dir else root / "static_controls_static_weak_v1"
    out = Path(args.out) if args.out else root

    knn = read_csv(root / "stageb6_knn_sensitivity.csv")
    lit = read_csv(root / "stageb6_literature_metrics.csv")
    residual = read_csv(static_dir / "stageb6_static_residualized_knn.csv")
    conditioned = read_csv(static_dir / "stageb6_static_conditioned_knn.csv")
    residual_lit = read_csv(static_dir / "stageb6_static_control_literature_metrics.csv")
    residual_diag = read_csv(static_dir / "stageb6_static_residual_diagnostics.csv")

    controls = {
        "action_effect": "action_effect_heldout_signature",
        "static": "static_heldout_signature",
        "shuffled": "action_column_shuffled_heldout",
    }
    knn_map: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in knn:
        if row.get("k") != "10" or float(row.get("jitter_epsilon", "nan")) != 0.0:
            continue
        if row.get("representation_components") != "32":
            continue
        pair = pair_key(row)
        for label, control in controls.items():
            if row.get("control") == control:
                knn_map[(pair, label)].append(f(row, "chance_adjusted_overlap"))

    lit_map: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in lit:
        if row.get("measurement") != "cknna_linear_cka":
            continue
        if row.get("representation_components") != "32":
            continue
        pair = pair_key(row)
        for label, control in controls.items():
            if row.get("control") == control:
                lit_map[(pair, label)].append(f(row, "calibrated_score"))

    res_map: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in residual:
        pair = pair_key(row)
        res_map[(pair, row["control"])].append(f(row, "chance_adjusted_overlap"))

    res_lit_map: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in residual_lit:
        if row.get("measurement") != "cknna_linear_cka":
            continue
        pair = pair_key(row)
        res_lit_map[(pair, row["control"])].append(f(row, "null_calibrated_score"))

    bins: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in conditioned:
        pair = pair_key(row)
        bins[(pair, row["static_bin_label"], row["control"])]["adjusted"].append(f(row, "chance_adjusted_overlap"))

    residual_energy: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in residual_diag:
        if row.get("control") == "static_residualized_probefit":
            residual_energy[(row["channel"], row["control"])].append(f(row, "residual_norm_fraction"))

    pairs = sorted({key[0] for key in knn_map} | {key[0] for key in res_map})
    rows: list[dict[str, object]] = []
    for pair in pairs:
        action_vals = knn_map[(pair, "action_effect")]
        static_vals = knn_map[(pair, "static")]
        shuffled_vals = knn_map[(pair, "shuffled")]
        res_vals = res_map[(pair, "static_residualized_probefit")]
        res_shuf_vals = res_map[(pair, "static_residualized_shuffled_probefit")]
        row: dict[str, object] = {
            "pair": pair,
            "n_action": len(action_vals),
            "action_effect_adjusted_mean": avg(action_vals),
            "action_effect_min": min_finite(action_vals),
            "action_effect_positive": count_pos(action_vals),
            "static_adjusted_mean": avg(static_vals),
            "shuffled_adjusted_mean": avg(shuffled_vals),
            "action_minus_static_mean": avg(action_vals) - avg(static_vals),
            "action_minus_shuffled_mean": avg(action_vals) - avg(shuffled_vals),
            "residualized_action_effect_mean": avg(res_vals),
            "residualized_action_min": min_finite(res_vals),
            "residualized_action_positive": count_pos(res_vals),
            "residualized_shuffled_mean": avg(res_shuf_vals),
            "residualized_minus_shuffled_mean": avg(res_vals) - avg(res_shuf_vals),
            "cknna_action_effect_mean": avg(lit_map[(pair, "action_effect")]),
            "cknna_static_mean": avg(lit_map[(pair, "static")]),
            "cknna_shuffled_mean": avg(lit_map[(pair, "shuffled")]),
            "cknna_residualized_mean": avg(res_lit_map[(pair, "static_residualized_probefit")]),
            "cknna_residualized_shuffled_mean": avg(res_lit_map[(pair, "static_residualized_shuffled_probefit")]),
        }
        for label in ("lowest", "low_mid", "high_mid", "highest"):
            ae = avg(bins[(pair, label, "action_effect_static_conditioned")]["adjusted"])
            sh = avg(bins[(pair, label, "shuffled_static_conditioned")]["adjusted"])
            row[f"conditioned_{label}_mean"] = ae
            row[f"conditioned_{label}_minus_shuffled"] = ae - sh
        row["conditioned_all_bins_positive"] = all(
            float(row[f"conditioned_{label}_mean"]) > 0
            and float(row[f"conditioned_{label}_minus_shuffled"]) > 0
            for label in ("lowest", "low_mid", "high_mid", "highest")
        )
        channels = pair.split(":")
        mins = []
        for channel in channels:
            vals = residual_energy[(channel, "static_residualized_probefit")]
            if vals:
                mins.append(min(vals))
                row[f"residual_norm_min_{channel}"] = min(vals)
                row[f"residual_norm_mean_{channel}"] = avg(vals)
        row["residual_norm_min_pair"] = min(mins) if mins else float("nan")
        row["candidate_generating_screen"] = bool(
            len(action_vals) == 9
            and row["residualized_action_effect_mean"] >= 0.03
            and row["residualized_minus_shuffled_mean"] >= 0.03
            and row["residualized_shuffled_mean"] <= 0.01
            and row["cknna_residualized_mean"] >= 0.02
            and row["residualized_action_positive"] == 9
            and row["conditioned_all_bins_positive"]
            and (np.isfinite(float(row["residual_norm_min_pair"])) and float(row["residual_norm_min_pair"]) >= 0.10)
        )
        rows.append(row)

    write_csv(out / "static_weak_pair_summary.csv", rows)
    print(f"Wrote {out / 'static_weak_pair_summary.csv'} rows={len(rows)}")


if __name__ == "__main__":
    main()
