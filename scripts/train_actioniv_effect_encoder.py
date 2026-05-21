#!/usr/bin/env python
"""Minimal Action-IV effect-subspace prototype.

This script is intentionally NumPy-only.  It evaluates whether paired cross-channel
action effects can identify a shared low-dimensional effect subspace.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from sae_align.action_iv.metrics import (
    fit_logistic,
    l2_normalize,
    metric_dict,
    predict_logistic,
    random_project,
    recall_at_k,
    train_val_test_by_state_with_labels,
    standardize_train_apply,
    choose_threshold,
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


def flatten_project(data: Mapping[str, np.ndarray], key: str, dim: int, seed: int) -> np.ndarray:
    arr = require(data, key)
    proj, _ = random_project(arr, dim, seed)
    return proj.astype(np.float32)


def fit_svd_effect_subspace(x_train: np.ndarray, y_train: np.ndarray, latent_dim: int):
    x_train_std, _, mx, sx = standardize_train_apply(x_train, x_train)
    y_train_std, _, my, sy = standardize_train_apply(y_train, y_train)
    cxy = (x_train_std.T @ y_train_std) / max(1, x_train_std.shape[0] - 1)
    u, _, vt = np.linalg.svd(cxy, full_matrices=False)
    return {
        "mx": mx.astype(np.float32),
        "sx": sx.astype(np.float32),
        "my": my.astype(np.float32),
        "sy": sy.astype(np.float32),
        "ux": u[:, :latent_dim].astype(np.float32),
        "vy": vt.T[:, :latent_dim].astype(np.float32),
    }


def apply_svd(model: Mapping[str, np.ndarray], x: np.ndarray, y: np.ndarray):
    xs = (x - model["mx"]) / model["sx"]
    ys = (y - model["my"]) / model["sy"]
    zx = xs @ model["ux"]
    zy = ys @ model["vy"]
    return l2_normalize(zx), l2_normalize(zy)


def derangement(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    if n <= 1:
        return np.arange(n)
    for _ in range(100):
        perm = rng.permutation(n)
        if np.all(perm != np.arange(n)):
            return perm
    return np.roll(np.arange(n), 1)


def recall_at_k_group(query: np.ndarray, target: np.ndarray, query_group: np.ndarray, target_group: np.ndarray, k: int) -> float:
    """Recall@k with any target row in the same group counted as positive."""
    q = l2_normalize(query)
    t = l2_normalize(target)
    sim = q @ t.T
    kk = min(int(k), sim.shape[1])
    top = np.argpartition(-sim, kth=max(0, kk - 1), axis=1)[:, :kk]
    qg = np.asarray(query_group)
    tg = np.asarray(target_group)
    hits = np.array([np.any(tg[top[i]] == qg[i]) for i in range(sim.shape[0])], dtype=np.float32)
    return float(np.mean(hits))


def bidirectional(row: Mapping[str, object]) -> float:
    return 0.5 * (float(row["recall_a_to_b"]) + float(row["recall_b_to_a"]))


def evaluate_task_head(z: np.ndarray, y: np.ndarray, state_ids: np.ndarray, seed: int):
    train, val, test = train_val_test_by_state_with_labels(state_ids, y, seed)
    z_train, z_all, _, _ = standardize_train_apply(z[train], z)
    theta, bias = fit_logistic(z_train, y[train], np.ones(np.sum(train), dtype=np.float32), epochs=120, lr=0.2, l2=1e-4)
    val_score = predict_logistic(z_all[val], theta, bias)
    test_score = predict_logistic(z_all[test], theta, bias)
    threshold = choose_threshold(y[val], val_score)
    out = {}
    out.update(metric_dict("val", y[val], val_score, threshold))
    out.update(metric_dict("test", y[test], test_score, threshold))
    return out


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    ensure_dir(out / "reports")
    data_npz = np.load(args.data, allow_pickle=True)
    data: Dict[str, np.ndarray] = {k: data_npz[k] for k in data_npz.files}
    ch_a, ch_b = args.channels
    state_ids = require(data, "state_id").astype(np.int64)
    y = require(data, "target_task_success").astype(np.float32).reshape(-1)

    xa = flatten_project(data, f"delta_{ch_a}", args.input_dim, args.seed + 1)
    yb = flatten_project(data, f"delta_{ch_b}", args.input_dim, args.seed + 2)
    static_a = flatten_project(data, f"obs0_{ch_a}", args.input_dim, args.seed + 3)
    static_b = flatten_project(data, f"obs0_{ch_b}", args.input_dim, args.seed + 4)

    train, val, test = train_val_test_by_state_with_labels(state_ids, y, args.seed)

    retrieval_rows: List[Dict[str, object]] = []
    task_rows: List[Dict[str, object]] = []

    for latent_dim in args.latent_dims:
        model = fit_svd_effect_subspace(xa[train], yb[train], latent_dim)
        za, zb = apply_svd(model, xa, yb)
        # Static SVD baseline with same machinery.
        static_model = fit_svd_effect_subspace(static_a[train], static_b[train], latent_dim)
        sa, sb = apply_svd(static_model, static_a, static_b)
        train_idx = np.flatnonzero(train)
        shuf_train_idx = train_idx[derangement(train_idx.size, args.seed + 5000 + int(latent_dim))]
        shuf_model = fit_svd_effect_subspace(xa[train], yb[shuf_train_idx], latent_dim)
        shza, shzb = apply_svd(shuf_model, xa, yb)
        xa_std_all = standardize_train_apply(xa[train], xa)[1]
        yb_std_all = standardize_train_apply(yb[train], yb)[1]

        for split_name, mask in [("train", train), ("val", val), ("test", test)]:
            for k in args.k_values:
                retrieval_rows.append({
                    "split": split_name,
                    "representation": "action_iv_svd",
                    "latent_dim": int(latent_dim),
                    "k": int(k),
                    "recall_a_to_b": recall_at_k(za[mask], zb[mask], int(k)),
                    "recall_b_to_a": recall_at_k(zb[mask], za[mask], int(k)),
                })
                retrieval_rows.append({
                    "split": split_name,
                    "representation": "static_svd",
                    "latent_dim": int(latent_dim),
                    "k": int(k),
                    "recall_a_to_b": recall_at_k_group(sa[mask], sb[mask], state_ids[mask], state_ids[mask], int(k)),
                    "recall_b_to_a": recall_at_k_group(sb[mask], sa[mask], state_ids[mask], state_ids[mask], int(k)),
                })
                retrieval_rows.append({
                    "split": split_name,
                    "representation": "raw_delta",
                    "latent_dim": int(args.input_dim),
                    "k": int(k),
                    "recall_a_to_b": recall_at_k(xa_std_all[mask], yb_std_all[mask], int(k)),
                    "recall_b_to_a": recall_at_k(yb_std_all[mask], xa_std_all[mask], int(k)),
                })
                perm = derangement(int(np.sum(mask)), args.seed + 99 + int(latent_dim) + int(k))
                retrieval_rows.append({
                    "split": split_name,
                    "representation": "shuffled_effect_control",
                    "latent_dim": int(latent_dim),
                    "k": int(k),
                    "recall_a_to_b": recall_at_k(shza[mask], shzb[mask][perm], int(k)),
                    "recall_b_to_a": recall_at_k(shzb[mask], shza[mask][perm], int(k)),
                })

        z_task = 0.5 * (za + zb)
        task = evaluate_task_head(z_task, y, state_ids, args.seed)
        task.update({"representation": "action_iv_svd", "latent_dim": int(latent_dim)})
        task_rows.append(task)
        s_task = 0.5 * (sa + sb)
        stask = evaluate_task_head(s_task, y, state_ids, args.seed)
        stask.update({"representation": "static_svd", "latent_dim": int(latent_dim)})
        task_rows.append(stask)
        # action-only baseline
        if latent_dim == args.latent_dims[0]:
            action = require(data, "action_array").astype(np.float32)
            abase = evaluate_task_head(action, y, state_ids, args.seed)
            abase.update({"representation": "action_only", "latent_dim": 0})
            task_rows.append(abase)
            rgb_base = evaluate_task_head(xa, y, state_ids, args.seed)
            rgb_base.update({"representation": f"raw_delta_{ch_a}", "latent_dim": int(args.input_dim)})
            task_rows.append(rgb_base)
            range_base = evaluate_task_head(yb, y, state_ids, args.seed)
            range_base.update({"representation": f"raw_delta_{ch_b}", "latent_dim": int(args.input_dim)})
            task_rows.append(range_base)

    write_csv(out / "reports" / "actioniv_retrieval.csv", retrieval_rows)
    write_csv(out / "reports" / "actioniv_task_head.csv", task_rows)

    # Decision summary: choose latent dim on validation, then evaluate test once.
    primary_k = 10
    val_rows = [r for r in retrieval_rows if r["split"] == "val" and r["k"] == primary_k]
    test_rows = [r for r in retrieval_rows if r["split"] == "test" and r["k"] == primary_k]
    val_by_dim: Dict[int, Dict[str, float]] = {}
    for latent_dim in args.latent_dims:
        rows_dim = [r for r in val_rows if int(r["latent_dim"]) == int(latent_dim) or (r["representation"] == "raw_delta" and int(r["latent_dim"]) == int(args.input_dim))]
        vals = {str(r["representation"]): bidirectional(r) for r in rows_dim}
        if "action_iv_svd" in vals:
            baseline = max(vals.get("static_svd", 0.0), vals.get("raw_delta", 0.0), vals.get("shuffled_effect_control", 0.0))
            val_by_dim[int(latent_dim)] = {"delta": vals["action_iv_svd"] - baseline}
    selected_dim = max(val_by_dim, key=lambda d: val_by_dim[d]["delta"]) if val_by_dim else int(args.latent_dims[0])
    selected_test = {str(r["representation"]): bidirectional(r) for r in test_rows if int(r["latent_dim"]) == int(selected_dim)}
    raw_test = [r for r in test_rows if r["representation"] == "raw_delta"]
    if raw_test:
        selected_test["raw_delta"] = bidirectional(raw_test[0])
    best_iv = selected_test.get("action_iv_svd", 0.0)
    best_static = selected_test.get("static_svd", 0.0)
    best_raw = selected_test.get("raw_delta", 0.0)
    best_shuf = selected_test.get("shuffled_effect_control", 0.0)
    task_by_rep = {str(r["representation"]): float(r["test_auprc"]) for r in task_rows}
    action_iv_task = max([float(r["test_auprc"]) for r in task_rows if r["representation"] == "action_iv_svd" and int(r["latent_dim"]) == int(selected_dim)] + [float("nan")])
    action_only_task = task_by_rep.get("action_only", float("nan"))
    raw_task = max(task_by_rep.get(f"raw_delta_{ch_a}", float("nan")), task_by_rep.get(f"raw_delta_{ch_b}", float("nan")))
    task_gate_pass = bool(
        np.isfinite(action_iv_task)
        and np.isfinite(action_only_task)
        and np.isfinite(raw_task)
        and action_iv_task - action_only_task >= args.go_task_delta
        and action_iv_task - raw_task >= args.go_task_delta
    )
    summary = {
        "selection_metric": "val_bidirectional_recall10_delta",
        "selected_latent_dim": int(selected_dim),
        "best_action_iv_recall10": best_iv,
        "best_static_recall10": best_static,
        "best_raw_delta_recall10": best_raw,
        "best_shuffled_recall10": best_shuf,
        "delta_vs_static": best_iv - best_static,
        "delta_vs_raw_delta": best_iv - best_raw,
        "delta_vs_shuffled": best_iv - best_shuf,
        "retrieval_gate_pass": bool(
            (best_iv - best_static >= args.go_retrieval_delta)
            and (best_iv - best_raw >= args.go_retrieval_delta)
            and (best_iv - best_shuf >= args.go_retrieval_delta)
        ),
        "action_iv_task_auprc": action_iv_task,
        "action_only_task_auprc": action_only_task,
        "raw_effect_task_auprc": raw_task,
        "task_delta_vs_action_only": action_iv_task - action_only_task,
        "task_delta_vs_raw_effect": action_iv_task - raw_task,
        "task_gate_pass": task_gate_pass,
    }
    summary["actioniv_gate_pass"] = bool(summary["retrieval_gate_pass"] and summary["task_gate_pass"])
    write_json(out / "reports" / "actioniv_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--channels", nargs=2, default=["rgb", "range"])
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--latent-dims", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--go-retrieval-delta", type=float, default=0.10)
    parser.add_argument("--go-task-delta", type=float, default=0.05)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
