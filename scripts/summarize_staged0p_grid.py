#!/usr/bin/env python
"""Aggregate Stage D0' predictor-grounded observability runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2, sort_keys=True, allow_nan=False)


def json_safe(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if math.isfinite(x) else None
    return obj


def f(row: Mapping[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def finite_max(values: List[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return max(finite) if finite else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--expected-report-dirs", type=int, default=None)
    p.add_argument("--phase", choices=["d0p", "d0"], default="d0p")
    args = p.parse_args()

    out = args.out or args.root
    report_files = sorted(args.root.glob("**/reports/staged0p_results.csv"))
    if args.expected_report_dirs is not None and len(report_files) != args.expected_report_dirs:
        raise RuntimeError(
            f"Expected {args.expected_report_dirs} report dirs, found {len(report_files)} under {args.root}"
        )

    rows: List[Dict[str, object]] = []
    for path in report_files:
        for row in read_csv(path):
            rr: Dict[str, object] = dict(row)
            rr["report_dir"] = str(path.parent.parent)
            rows.append(rr)
    write_csv(out / "staged0p_grid_results.csv", rows)

    by_key: Dict[Tuple[str, float], List[Mapping[str, object]]] = {}
    for r in rows:
        method = str(r["method"])
        alpha = float(r["alpha"])
        by_key.setdefault((method, alpha), []).append(r)

    summary: List[Dict[str, object]] = []
    for (method, alpha), group in sorted(by_key.items()):
        vals = lambda key: np.array([f(g, key) for g in group], dtype=float)
        auprc_delta = vals("test_auprc_delta_vs_uniform")
        f1_delta = vals("test_f1_delta_vs_uniform")
        behavior_delta = np.maximum(auprc_delta, f1_delta)
        summary.append(
            {
                "method": method,
                "alpha": alpha,
                "n_rows": len(group),
                "test_auprc_delta_mean": float(np.nanmean(auprc_delta)),
                "test_auprc_delta_min": float(np.nanmin(auprc_delta)),
                "test_auprc_delta_positive_frac": float(np.nanmean(auprc_delta > 0)),
                "test_f1_delta_mean": float(np.nanmean(f1_delta)),
                "test_f1_delta_min": float(np.nanmin(f1_delta)),
                "test_f1_delta_positive_frac": float(np.nanmean(f1_delta > 0)),
                "test_behavior_delta_mean": float(np.nanmean(behavior_delta)),
                "test_behavior_delta_min": float(np.nanmin(behavior_delta)),
                "test_behavior_delta_positive_frac": float(np.nanmean(behavior_delta > 0)),
                "passes_plus_0p10_auprc": bool(np.nanmean(auprc_delta) >= 0.10),
                "passes_plus_0p05_auprc": bool(np.nanmean(auprc_delta) >= 0.05),
                "passes_plus_0p10_behavior": bool(np.nanmean(behavior_delta) >= 0.10),
                "passes_plus_0p05_behavior": bool(np.nanmean(behavior_delta) >= 0.05),
            }
        )
    write_csv(out / "staged0p_grid_summary.csv", summary)

    def best(method: str, key: str = "test_auprc_delta_mean") -> float:
        candidates = [r for r in summary if r["method"] == method]
        if not candidates:
            return float("nan")
        return max(float(r[key]) for r in candidates)

    def best_behavior(method: str) -> float:
        return best(method, "test_behavior_delta_mean")

    def best_row(method: str, key: str = "test_behavior_delta_mean") -> Mapping[str, object]:
        candidates = [r for r in summary if r["method"] == method]
        if not candidates:
            return {}
        return max(candidates, key=lambda r: float(r[key]))

    decision = {
        "n_report_dirs": len(report_files),
        "n_rows": len(rows),
        "oracle_event_best_auprc_delta": best("oracle_event"),
        "oracle_event_best_f1_delta": best("oracle_event", "test_f1_delta_mean"),
        "oracle_event_best_auprc_row": best_row("oracle_event", "test_auprc_delta_mean"),
        "oracle_event_best_f1_row": best_row("oracle_event", "test_f1_delta_mean"),
        "observability_best_auprc_delta": best("observability"),
        "observability_best_f1_delta": best("observability", "test_f1_delta_mean"),
        "observability_best_auprc_row": best_row("observability", "test_auprc_delta_mean"),
        "observability_best_f1_row": best_row("observability", "test_f1_delta_mean"),
        "predictor_grounded_best_auprc_delta": best("predictor_grounded"),
        "predictor_grounded_best_f1_delta": best("predictor_grounded", "test_f1_delta_mean"),
        "predictor_uncertainty_best_auprc_delta": best("predictor_uncertainty"),
        "lossgrad_observability_best_auprc_delta": best("lossgrad_observability"),
        "shuffled_observability_best_auprc_delta": best("shuffled_observability"),
        "shuffled_observability_best_f1_delta": best("shuffled_observability", "test_f1_delta_mean"),
        "shuffled_predictor_grounded_best_auprc_delta": best("shuffled_predictor_grounded"),
        "shuffled_predictor_grounded_best_f1_delta": best("shuffled_predictor_grounded", "test_f1_delta_mean"),
        "change_mask_best_auprc_delta": best("change_mask"),
        "change_mask_best_f1_delta": best("change_mask", "test_f1_delta_mean"),
        "oracle_event_best_behavior_delta": best_behavior("oracle_event"),
        "observability_best_behavior_delta": best_behavior("observability"),
        "predictor_grounded_best_behavior_delta": best_behavior("predictor_grounded"),
        "predictor_uncertainty_best_behavior_delta": best_behavior("predictor_uncertainty"),
        "lossgrad_observability_best_behavior_delta": best_behavior("lossgrad_observability"),
        "shuffled_observability_best_behavior_delta": best_behavior("shuffled_observability"),
        "shuffled_predictor_grounded_best_behavior_delta": best_behavior("shuffled_predictor_grounded"),
        "change_mask_best_behavior_delta": best_behavior("change_mask"),
        "oracle_event_best_behavior_row": best_row("oracle_event"),
        "predictor_grounded_best_auprc_row": best_row("predictor_grounded", "test_auprc_delta_mean"),
        "predictor_grounded_best_f1_row": best_row("predictor_grounded", "test_f1_delta_mean"),
        "predictor_grounded_best_behavior_row": best_row("predictor_grounded"),
        "stageb_reference_signal": 0.05,
    }
    oracle_auprc_best = decision["oracle_event_best_auprc_row"]
    oracle_f1_best = decision["oracle_event_best_f1_row"]
    oracle_auprc_pass = bool(
        decision["oracle_event_best_auprc_delta"] >= 0.10
        and float(oracle_auprc_best.get("test_auprc_delta_positive_frac", 0.0)) >= 0.80
    )
    oracle_f1_pass = bool(
        decision["oracle_event_best_f1_delta"] >= 0.10
        and float(oracle_f1_best.get("test_f1_delta_positive_frac", 0.0)) >= 0.80
    )
    oracle_pass = bool(oracle_auprc_pass or oracle_f1_pass)
    best_control = finite_max(
        [
            decision["change_mask_best_behavior_delta"],
            decision["observability_best_behavior_delta"],
            decision["shuffled_observability_best_behavior_delta"],
            decision["shuffled_predictor_grounded_best_behavior_delta"],
        ]
    )
    best_control_auprc = finite_max(
        [
            decision["change_mask_best_auprc_delta"],
            decision["observability_best_auprc_delta"],
            decision["shuffled_observability_best_auprc_delta"],
            decision["shuffled_predictor_grounded_best_auprc_delta"],
        ]
    )
    best_control_f1 = finite_max(
        [
            decision["change_mask_best_f1_delta"],
            decision["observability_best_f1_delta"],
            decision["shuffled_observability_best_f1_delta"],
            decision["shuffled_predictor_grounded_best_f1_delta"],
        ]
    )
    pg_auprc_best = decision["predictor_grounded_best_auprc_row"]
    pg_f1_best = decision["predictor_grounded_best_f1_row"]
    pg_auprc_pass_raw = bool(
        decision["predictor_grounded_best_auprc_delta"] >= 0.10
        and decision["predictor_grounded_best_auprc_delta"] > best_control_auprc
        and float(pg_auprc_best.get("test_auprc_delta_positive_frac", 0.0)) >= 0.80
    )
    pg_f1_pass_raw = bool(
        decision["predictor_grounded_best_f1_delta"] >= 0.10
        and decision["predictor_grounded_best_f1_delta"] > best_control_f1
        and float(pg_f1_best.get("test_f1_delta_positive_frac", 0.0)) >= 0.80
    )
    pg_pass_raw = bool(pg_auprc_pass_raw or pg_f1_pass_raw)
    obs_auprc_best = decision["observability_best_auprc_row"]
    obs_f1_best = decision["observability_best_f1_row"]
    obs_control_auprc = finite_max(
        [
            decision["change_mask_best_auprc_delta"],
            decision["shuffled_observability_best_auprc_delta"],
        ]
    )
    obs_control_f1 = finite_max(
        [
            decision["change_mask_best_f1_delta"],
            decision["shuffled_observability_best_f1_delta"],
        ]
    )
    obs_auprc_pass_raw = bool(
        decision["observability_best_auprc_delta"] >= 0.10
        and decision["observability_best_auprc_delta"] > obs_control_auprc
        and float(obs_auprc_best.get("test_auprc_delta_positive_frac", 0.0)) >= 0.80
    )
    obs_f1_pass_raw = bool(
        decision["observability_best_f1_delta"] >= 0.10
        and decision["observability_best_f1_delta"] > obs_control_f1
        and float(obs_f1_best.get("test_f1_delta_positive_frac", 0.0)) >= 0.80
    )
    obs_pass_raw = bool(obs_auprc_pass_raw or obs_f1_pass_raw)
    proposed_pass_raw = bool(pg_pass_raw or (args.phase == "d0" and obs_pass_raw))
    proposed_pass = bool(oracle_pass and proposed_pass_raw)
    if oracle_pass and proposed_pass:
        branch = "branch_1_best_case_proceed_to_c1" if args.phase == "d0p" else "branch_1_oracle_and_observability_pass_proceed_to_c1"
    elif oracle_pass and not proposed_pass:
        branch = "branch_2_stop_toy_or_environment_migration" if args.phase == "d0p" else "branch_2_oracle_pass_observability_fail_stop_environment"
    else:
        branch = "branch_3_detector_failed_stop_toy" if args.phase == "d0p" else "branch_3_oracle_failed_stop_environment"
    decision.update(
        {
            "phase": args.phase,
            "oracle_pass_plus_0p10": oracle_pass,
            "oracle_auprc_pass_plus_0p10": oracle_auprc_pass,
            "oracle_f1_pass_plus_0p10": oracle_f1_pass,
            "observability_raw_auprc_pass_plus_0p10_and_controls": obs_auprc_pass_raw,
            "observability_raw_f1_pass_plus_0p10_and_controls": obs_f1_pass_raw,
            "observability_raw_pass_plus_0p10_and_controls": obs_pass_raw,
            "predictor_grounded_raw_auprc_pass_plus_0p10_and_controls": pg_auprc_pass_raw,
            "predictor_grounded_raw_f1_pass_plus_0p10_and_controls": pg_f1_pass_raw,
            "predictor_grounded_raw_pass_plus_0p10_and_controls": pg_pass_raw,
            "predictor_grounded_pass_plus_0p10_and_controls": bool(oracle_pass and pg_pass_raw),
            "proposed_pass_plus_0p10_and_controls": proposed_pass,
            "best_control_behavior_delta": best_control,
            "best_control_auprc_delta": best_control_auprc,
            "best_control_f1_delta": best_control_f1,
            "observability_auprc_control_margin": decision["observability_best_auprc_delta"] - obs_control_auprc,
            "observability_f1_control_margin": decision["observability_best_f1_delta"] - obs_control_f1,
            "predictor_grounded_control_margin": decision["predictor_grounded_best_behavior_delta"] - best_control,
            "predictor_grounded_auprc_control_margin": decision["predictor_grounded_best_auprc_delta"] - best_control_auprc,
            "predictor_grounded_f1_control_margin": decision["predictor_grounded_best_f1_delta"] - best_control_f1,
            "predictor_grounded_interpretable": bool(oracle_pass),
            "decision_branch": branch,
        }
    )
    write_json(out / "staged0p_decision_grid.json", decision)


if __name__ == "__main__":
    main()
