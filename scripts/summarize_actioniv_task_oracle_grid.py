#!/usr/bin/env python
"""Summarize Action-IV Step-2 official-task oracle array outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)


def write_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    ensure_dir(path.parent)
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
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--expected-report-dirs", type=int, default=None)
    parser.add_argument("--go-delta", type=float, default=0.10)
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root
    rows: List[Dict[str, object]] = []
    decision_paths = sorted(root.glob("seed_*/task_oracle/reports/actioniv_task_oracle_decision.json"))
    single_decision = root / "task_oracle" / "reports" / "actioniv_task_oracle_decision.json"
    if single_decision.exists():
        decision_paths.append(single_decision)
    for decision_path in decision_paths:
        run_dir = decision_path.parents[2]
        seed_name = run_dir.name if run_dir != root else "single"
        try:
            decision = read_json(decision_path)
        except json.JSONDecodeError:
            rows.append({
                "run_dir": str(run_dir),
                "seed_name": seed_name,
                "decision_branch": "invalid_unreadable_decision_json",
            })
            continue
        summary_path = run_dir / "task_oracle" / "reports" / "actioniv_task_oracle_summary.csv"
        if summary_path.exists():
            with open(summary_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    row_out: Dict[str, object] = {"run_dir": str(run_dir), "seed_name": seed_name}
                    row_out.update(row)
                    row_out["decision_branch"] = decision.get("branch", "")
                    rows.append(row_out)
        else:
            rows.append({
                "run_dir": str(run_dir),
                "seed_name": seed_name,
                "decision_branch": decision.get("branch", ""),
                "oracle_mean_auprc_delta": decision.get("oracle_mean_auprc_delta", np.nan),
            })

    report_dirs = sorted({row["run_dir"] for row in rows})
    if args.expected_report_dirs is not None and len(report_dirs) != int(args.expected_report_dirs):
        raise SystemExit(f"Expected {args.expected_report_dirs} report dirs, found {len(report_dirs)}")

    write_csv(out / "actioniv_task_oracle_grid_rows.csv", rows)
    oracle_rows = [r for r in rows if r.get("method") == "oracle_task"]
    deltas = np.array([float(r["auprc_delta_vs_uniform"]) for r in oracle_rows], dtype=np.float64)
    finite = deltas[np.isfinite(deltas)]
    mean_delta = float(np.mean(finite)) if finite.size else None
    min_delta = float(np.min(finite)) if finite.size else None
    positive_fraction = float(np.mean(finite > 0.0)) if finite.size else 0.0
    pass_gate = bool(finite.size > 0 and mean_delta is not None and min_delta is not None and mean_delta >= float(args.go_delta) and min_delta > 0.0)
    invalid_decisions = []
    for decision_path in decision_paths:
        try:
            decision = read_json(decision_path)
        except json.JSONDecodeError:
            invalid_decisions.append(str(decision_path))
            continue
        if decision.get("branch") == "invalid_degenerate_target":
            invalid_decisions.append(str(decision_path))
    branch = "invalid_degenerate_target" if invalid_decisions else ("oracle_pass" if pass_gate else "oracle_failed")
    summary = {
        "branch": branch,
        "go_delta": float(args.go_delta),
        "n_report_dirs": int(len(report_dirs)),
        "n_rows": int(len(rows)),
        "n_oracle_rows": int(len(oracle_rows)),
        "oracle_mean_auprc_delta": mean_delta,
        "oracle_min_auprc_delta": min_delta,
        "oracle_positive_fraction": positive_fraction,
        "primary_metric": "validation-selected test AUPRC delta",
        "invalid_decision_files": invalid_decisions,
    }
    write_json(out / "actioniv_task_oracle_grid_decision.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
