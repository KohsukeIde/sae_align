#!/usr/bin/env python
"""Official-task oracle sanity for the Action-IV phase.

This is Step 2.  It asks only whether a task-defined oracle signal can beat uniform.
It is not a PSP/Dreamer comparison and not the Action-IV prototype itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from sae_align.action_iv.metrics import (
    average_rank,
    choose_threshold,
    fit_logistic,
    metric_dict,
    predict_logistic,
    random_project,
    rank_train_apply,
    train_val_test_by_state_with_labels,
    standardize_train_apply,
)


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
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def require(data: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing key {key!r}. Available keys: {sorted(data.keys())[:50]}")
    return np.asarray(data[key])


def flatten_obs(data: Mapping[str, np.ndarray], channel: str, dim: int, seed: int) -> np.ndarray:
    arr = require(data, f"obs0_{channel}")
    proj, _ = random_project(arr, dim, seed)
    return proj.astype(np.float32)


def one_hot_strings(values: np.ndarray) -> np.ndarray:
    vals = values.astype(str)
    classes = sorted(np.unique(vals).tolist())
    out = np.zeros((vals.size, len(classes)), dtype=np.float32)
    idx = {c: i for i, c in enumerate(classes)}
    for i, v in enumerate(vals):
        out[i, idx[v]] = 1.0
    return out


def build_features(data: Mapping[str, np.ndarray], channels: Sequence[str], dim: int, seed: int, train_mask: np.ndarray) -> np.ndarray:
    parts: List[np.ndarray] = []
    for ci, ch in enumerate(channels):
        x = flatten_obs(data, ch, dim, seed + 1009 * ci)
        _, x_all, _, _ = standardize_train_apply(x[train_mask], x)
        parts.append(x_all)
    action = require(data, "action_array").astype(np.float32)
    _, action_all, _, _ = standardize_train_apply(action[train_mask], action)
    parts.append(action_all)
    parts.append(one_hot_strings(require(data, "action_type")))
    return np.concatenate(parts, axis=1).astype(np.float32)


def score_vectors(data: Mapping[str, np.ndarray], train_mask: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    y = require(data, "target_task_success").astype(np.float32).reshape(-1)
    change = np.linalg.norm(require(data, "world_delta").reshape(y.size, -1), axis=1).astype(np.float32)
    obs_parts = []
    for ch in ("rgb", "range"):
        key = f"detect_{ch}"
        if key in data:
            vals = require(data, key).astype(np.float32).reshape(-1)
            obs_parts.append(rank_train_apply(vals[train_mask], vals))
    if obs_parts:
        obs = obs_parts[0]
        for p in obs_parts[1:]:
            obs = obs * np.maximum(p, 1e-8)
        obs = np.power(np.maximum(obs, 0.0), 1.0 / float(len(obs_parts))).astype(np.float32)
    else:
        obs = np.zeros_like(y)
    rng = np.random.default_rng(seed + 9173)
    shuffled_obs = obs.copy()
    rng.shuffle(shuffled_obs)
    return {
        "oracle_task": y,
        "change_mask": rank_train_apply(change[train_mask], change),
        "observability": obs,
        "shuffled_observability": shuffled_obs.astype(np.float32),
    }


def method_weights(method: str, alpha: float, scores: Dict[str, np.ndarray], train_mask: np.ndarray) -> np.ndarray:
    n = int(np.sum(train_mask))
    if method == "uniform":
        return np.ones(n, dtype=np.float32)
    if method == "oracle_task":
        y_train = scores["oracle_task"][train_mask]
        return np.where(y_train > 0.5, float(alpha), 1.0).astype(np.float32)
    if method not in scores:
        raise ValueError(f"Unknown method {method!r}")
    s = scores[method]
    return (1.0 + float(alpha) * np.clip(s[train_mask], 0.0, 1.0)).astype(np.float32)


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    ensure_dir(out / "reports")
    data_npz = np.load(args.data, allow_pickle=True)
    data: Dict[str, np.ndarray] = {k: data_npz[k] for k in data_npz.files}
    y = require(data, "target_task_success").astype(np.float32).reshape(-1)
    state_ids = require(data, "state_id").astype(np.int64)
    positive_rate = float(np.mean(y > 0.5))
    if positive_rate <= 0.0 or positive_rate >= 1.0:
        write_json(out / "reports" / "actioniv_task_oracle_decision.json", {
            "branch": "invalid_degenerate_target",
            "go_delta": float(args.go_delta),
            "primary_metric": "test_auprc_after_val_auprc_selection",
            "target_positive_rate": positive_rate,
            "n_rows": int(y.size),
            "n_selected_rows": 0,
            "oracle_mean_auprc_delta": None,
            "oracle_min_auprc_delta": None,
        })
        print(json.dumps({"branch": "invalid_degenerate_target", "target_positive_rate": positive_rate}, indent=2))
        return

    rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    for split_seed in args.split_seeds:
        train_mask, val_mask, test_mask = train_val_test_by_state_with_labels(state_ids, y, int(split_seed))
        X = build_features(data, args.input_channels, args.channel_dim, int(split_seed), train_mask)
        X_train, X_all, _, _ = standardize_train_apply(X[train_mask], X)
        X_val = X_all[val_mask]
        X_test = X_all[test_mask]
        y_train = y[train_mask]
        y_val = y[val_mask]
        y_test = y[test_mask]
        scores = score_vectors(data, train_mask, int(split_seed))
        by_method: Dict[str, List[Dict[str, object]]] = {}
        for method in args.methods:
            for alpha in args.alphas:
                weights = method_weights(method, float(alpha), scores, train_mask)
                theta, bias = fit_logistic(X_train, y_train, weights, epochs=args.epochs, lr=args.lr, l2=args.l2)
                val_score = predict_logistic(X_val, theta, bias)
                test_score = predict_logistic(X_test, theta, bias)
                threshold = choose_threshold(y_val, val_score)
                row: Dict[str, object] = {
                    "split_seed": int(split_seed),
                    "method": method,
                    "alpha": float(alpha),
                    "n_train": int(np.sum(train_mask)),
                    "n_val": int(np.sum(val_mask)),
                    "n_test": int(np.sum(test_mask)),
                    "train_positive_rate": float(np.mean(y_train)),
                    "test_positive_rate": float(np.mean(y_test)),
                }
                row.update(metric_dict("val", y_val, val_score, threshold))
                row.update(metric_dict("test", y_test, test_score, threshold))
                rows.append(row)
                by_method.setdefault(method, []).append(row)

        for method, rs in by_method.items():
            # Alpha is selected on validation AUPRC only; test is reported once.
            selected = max(rs, key=lambda r: np.nan_to_num(float(r["val_auprc"]), nan=-np.inf))
            selected = dict(selected)
            selected["selection_metric"] = "val_auprc"
            selected_rows.append(selected)

    write_csv(out / "reports" / "actioniv_task_oracle_results.csv", rows)
    write_csv(out / "reports" / "actioniv_task_oracle_selected.csv", selected_rows)
    # Aggregate validation-selected deltas vs validation-selected uniform.
    by_seed: Dict[int, Dict[str, Dict[str, object]]] = {}
    for row in selected_rows:
        by_seed.setdefault(int(row["split_seed"]), {}).setdefault(str(row["method"]), []).append(row)
    summary_rows: List[Dict[str, object]] = []
    for seed, method_rows in by_seed.items():
        uniform_rows = method_rows.get("uniform", [])
        if not uniform_rows:
            continue
        uniform = uniform_rows[0]
        uniform_auprc = float(uniform["test_auprc"])
        uniform_f1 = float(uniform["test_f1"])
        for method, rs in method_rows.items():
            selected = rs[0]
            summary_rows.append({
                "split_seed": seed,
                "method": method,
                "selected_alpha": float(selected["alpha"]),
                "selected_val_auprc": float(selected["val_auprc"]),
                "selected_test_auprc": float(selected["test_auprc"]),
                "selected_test_f1": float(selected["test_f1"]),
                "auprc_delta_vs_uniform": float(selected["test_auprc"]) - uniform_auprc,
                "f1_delta_vs_uniform": float(selected["test_f1"]) - uniform_f1,
            })
    write_csv(out / "reports" / "actioniv_task_oracle_summary.csv", summary_rows)
    oracle_deltas = [r for r in summary_rows if r["method"] == "oracle_task"]
    oracle_vals = [float(r["auprc_delta_vs_uniform"]) for r in oracle_deltas if np.isfinite(float(r["auprc_delta_vs_uniform"]))]
    oracle_mean_delta = float(np.mean(oracle_vals)) if oracle_vals else float("-inf")
    oracle_min_delta = float(np.min(oracle_vals)) if oracle_vals else float("-inf")
    branch = "oracle_pass" if oracle_mean_delta >= args.go_delta and oracle_min_delta > 0.0 else "oracle_failed"
    write_json(out / "reports" / "actioniv_task_oracle_decision.json", {
        "branch": branch,
        "go_delta": float(args.go_delta),
        "primary_metric": "test_auprc_after_val_auprc_selection",
        "oracle_mean_auprc_delta": float(oracle_mean_delta),
        "oracle_min_auprc_delta": float(oracle_min_delta),
        "n_rows": len(rows),
        "n_selected_rows": len(selected_rows),
    })
    print(json.dumps({"branch": branch, "oracle_mean_auprc_delta": oracle_mean_delta}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--input-channels", nargs="+", default=["rgb"])
    parser.add_argument("--methods", nargs="+", default=["uniform", "oracle_task", "change_mask", "observability", "shuffled_observability"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--channel-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--go-delta", type=float, default=0.10)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
