#!/usr/bin/env python
"""Postmortem audit for the Action-IV Step-2 oracle gate.

This is not a new Action-IV result and does not advance to Step 3.  It audits
whether the precommitted oracle-task sample-weighting gate was an appropriate
upper-bound check for DestroyAll task learnability.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from sae_align.action_iv.metrics import (
    average_precision,
    choose_threshold,
    fit_logistic,
    metric_dict,
    predict_logistic,
    random_project,
    rank_train_apply,
    train_val_test_by_state,
    standardize_train_apply,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x: object) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
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


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def require(data: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing key {key!r}. Available keys: {sorted(data.keys())}")
    return np.asarray(data[key])


def one_hot_strings(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values).astype(str).reshape(-1)
    classes = sorted(np.unique(vals).tolist())
    idx = {c: i for i, c in enumerate(classes)}
    out = np.zeros((vals.size, len(classes)), dtype=np.float32)
    for i, value in enumerate(vals):
        out[i, idx[value]] = 1.0
    return out


def project_array(arr: np.ndarray, dim: int, seed: int) -> np.ndarray:
    proj, _ = random_project(np.asarray(arr, dtype=np.float32), int(dim), int(seed))
    return proj.astype(np.float32)


def action_features(data: Mapping[str, np.ndarray]) -> np.ndarray:
    parts = [require(data, "action_array").astype(np.float32), one_hot_strings(require(data, "action_type"))]
    return np.concatenate(parts, axis=1).astype(np.float32)


def obs_features(data: Mapping[str, np.ndarray], channels: Sequence[str], dim: int, seed: int) -> np.ndarray:
    parts = []
    for ci, channel in enumerate(channels):
        parts.append(project_array(require(data, f"obs0_{channel}"), dim, seed + 1009 * ci))
    return np.concatenate(parts, axis=1).astype(np.float32)


def world_delta_diag_features(data: Mapping[str, np.ndarray], dim: int, seed: int) -> np.ndarray:
    world_delta = require(data, "world_delta").astype(np.float32)
    projected = project_array(world_delta, dim, seed)
    norm = np.linalg.norm(world_delta.reshape(world_delta.shape[0], -1), axis=1, keepdims=True).astype(np.float32)
    reward_delta = require(data, "target_reward_delta").astype(np.float32).reshape(-1, 1)
    return np.concatenate([projected, norm, reward_delta], axis=1).astype(np.float32)


def build_feature_sets(data: Mapping[str, np.ndarray], channel_dim: int, seed: int) -> Dict[str, np.ndarray]:
    n = require(data, "target_task_success").shape[0]
    action = action_features(data)
    obs_rgb = obs_features(data, ["rgb"], channel_dim, seed + 11)
    obs_range = obs_features(data, ["range"], channel_dim, seed + 23)
    obs_rgb_range = np.concatenate([obs_rgb, obs_range], axis=1).astype(np.float32)
    world_diag = world_delta_diag_features(data, channel_dim, seed + 37)
    return {
        "intercept_only": np.ones((n, 1), dtype=np.float32),
        "action_only": action,
        "obs_rgb_only": obs_rgb,
        "obs_range_only": obs_range,
        "obs_rgb_range_only": obs_rgb_range,
        "obs_rgb_action": np.concatenate([obs_rgb, action], axis=1).astype(np.float32),
        "obs_rgb_range_action": np.concatenate([obs_rgb_range, action], axis=1).astype(np.float32),
        "obs_rgb_action_world_delta_diag": np.concatenate([obs_rgb, action, world_diag], axis=1).astype(np.float32),
        "obs_rgb_range_action_world_delta_diag": np.concatenate([obs_rgb_range, action, world_diag], axis=1).astype(np.float32),
    }


def train_eval_feature_set(
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
) -> Dict[str, float]:
    X_train_raw = X[train_mask]
    X_train, X_all, _, _ = standardize_train_apply(X_train_raw, X)
    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]
    weights = np.ones_like(y_train, dtype=np.float32)
    theta, bias = fit_logistic(X_train, y_train, weights, epochs=epochs, lr=lr, l2=l2)
    val_score = predict_logistic(X_all[val_mask], theta, bias)
    test_score = predict_logistic(X_all[test_mask], theta, bias)
    threshold = choose_threshold(y_val, val_score)
    out: Dict[str, float] = {}
    out.update(metric_dict("val", y_val, val_score, threshold))
    out.update(metric_dict("test", y_test, test_score, threshold))
    out["test_score_mean_pos"] = float(np.mean(test_score[y_test > 0.5])) if np.any(y_test > 0.5) else float("nan")
    out["test_score_mean_neg"] = float(np.mean(test_score[y_test <= 0.5])) if np.any(y_test <= 0.5) else float("nan")
    out["test_score_gap_pos_minus_neg"] = out["test_score_mean_pos"] - out["test_score_mean_neg"]
    return out


def split_with_labels_record(
    state_ids: np.ndarray,
    y: np.ndarray,
    seed: int,
    min_pos: int = 1,
    min_neg: int = 1,
    max_tries: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    labels = np.asarray(y).astype(bool).reshape(-1)
    for offset in range(int(max_tries)):
        actual_seed = int(seed) + int(offset)
        train_mask, val_mask, test_mask = train_val_test_by_state(state_ids, actual_seed)
        ok = True
        for mask in (train_mask, val_mask, test_mask):
            pos = int(np.sum(labels[mask]))
            neg = int(np.sum(~labels[mask]))
            if pos < int(min_pos) or neg < int(min_neg):
                ok = False
                break
        if ok:
            return train_mask, val_mask, test_mask, actual_seed, int(offset)
    train_mask, val_mask, test_mask = train_val_test_by_state(state_ids, int(seed))
    return train_mask, val_mask, test_mask, int(seed), -1


def oracle_as_feature_rows(
    base_X: np.ndarray,
    y: np.ndarray,
    state_ids: np.ndarray,
    split_seeds: Sequence[int],
    data_seed: int,
    epochs: int,
    lr: float,
    l2: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    X = np.concatenate([base_X, y.reshape(-1, 1).astype(np.float32)], axis=1).astype(np.float32)
    for split_seed in split_seeds:
        train_mask, val_mask, test_mask, actual_seed, retry_offset = split_with_labels_record(state_ids, y, int(split_seed))
        metrics = train_eval_feature_set(X, y, train_mask, val_mask, test_mask, epochs, lr, l2)
        row: Dict[str, object] = {
            "data_seed": data_seed,
            "split_seed": int(split_seed),
            "actual_split_seed": int(actual_seed),
            "split_retry_offset": int(retry_offset),
            "feature_set": "obs_rgb_action_plus_oracle_label",
            "n_train": int(np.sum(train_mask)),
            "n_val": int(np.sum(val_mask)),
            "n_test": int(np.sum(test_mask)),
            "train_positive_rate": float(np.mean(y[train_mask] > 0.5)),
            "val_positive_rate": float(np.mean(y[val_mask] > 0.5)),
            "test_positive_rate": float(np.mean(y[test_mask] > 0.5)),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def postmortem_score_vectors(data: Mapping[str, np.ndarray], train_mask: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    y = require(data, "target_task_success").astype(np.float32).reshape(-1)
    change = np.linalg.norm(require(data, "world_delta").reshape(y.size, -1), axis=1).astype(np.float32)
    obs_parts = []
    for channel in ("rgb", "range"):
        key = f"detect_{channel}"
        if key in data:
            vals = require(data, key).astype(np.float32).reshape(-1)
            obs_parts.append(rank_train_apply(vals[train_mask], vals))
    if obs_parts:
        obs = obs_parts[0]
        for part in obs_parts[1:]:
            obs = obs * np.maximum(part, 1e-8)
        obs = np.power(np.maximum(obs, 0.0), 1.0 / float(len(obs_parts))).astype(np.float32)
    else:
        obs = np.zeros_like(y)
    rng = np.random.default_rng(int(seed) + 9173)
    shuffled_obs = obs.copy()
    shuffled_train = obs[train_mask].copy()
    rng.shuffle(shuffled_train)
    shuffled_obs[train_mask] = shuffled_train
    return {
        "oracle_task": y,
        "change_mask": rank_train_apply(change[train_mask], change),
        "observability": obs,
        "shuffled_observability": shuffled_obs.astype(np.float32),
    }


def postmortem_weights(method: str, alpha: float, scores: Mapping[str, np.ndarray], train_mask: np.ndarray) -> np.ndarray:
    n_train = int(np.sum(train_mask))
    if method == "uniform":
        return np.ones(n_train, dtype=np.float32)
    if method == "oracle_task":
        y_train = scores["oracle_task"][train_mask]
        return np.where(y_train > 0.5, float(alpha), 1.0).astype(np.float32)
    if method not in scores:
        return np.ones(n_train, dtype=np.float32)
    s = scores[method]
    return (1.0 + float(alpha) * np.clip(s[train_mask], 0.0, 1.0)).astype(np.float32)


def weight_diagnostics(weights: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    y = np.asarray(y_train, dtype=bool).reshape(-1)
    total = float(np.sum(w)) + 1e-12
    ess = float((np.sum(w) ** 2) / (np.sum(w * w) + 1e-12))
    n = max(1, w.size)
    top_n = max(1, int(np.ceil(0.1 * n)))
    top_idx = np.argsort(-w, kind="mergesort")[:top_n]
    return {
        "weight_mean": float(np.mean(w)),
        "weight_std": float(np.std(w)),
        "weight_min": float(np.min(w)),
        "weight_max": float(np.max(w)),
        "weight_ess": ess,
        "weight_ess_fraction": ess / float(n),
        "top_decile_weight_mass": float(np.sum(w[top_idx]) / total),
        "positive_train_rate": float(np.mean(y)),
        "positive_weight_mass": float(np.sum(w[y]) / total) if np.any(y) else 0.0,
        "negative_weight_mass": float(np.sum(w[~y]) / total) if np.any(~y) else 0.0,
    }


def selected_weighting_breakdown(
    report_dir: Path,
    data_seed: int,
    data: Mapping[str, np.ndarray],
    split_seeds: Sequence[int],
) -> List[Dict[str, object]]:
    selected_path = report_dir / "actioniv_task_oracle_selected.csv"
    if not selected_path.exists():
        return []
    selected = read_csv(selected_path)
    uniform_by_split = {int(r["split_seed"]): r for r in selected if r.get("method") == "uniform"}
    y = require(data, "target_task_success").astype(np.float32).reshape(-1)
    state_ids = require(data, "state_id").astype(np.int64)
    split_info: Dict[int, Tuple[np.ndarray, int, int]] = {}
    for split_seed in split_seeds:
        train_mask, _, _, actual_seed, retry_offset = split_with_labels_record(state_ids, y, int(split_seed))
        split_info[int(split_seed)] = (train_mask, actual_seed, retry_offset)
    rows: List[Dict[str, object]] = []
    for row in selected:
        split_seed = int(row["split_seed"])
        uniform = uniform_by_split.get(split_seed)
        if split_seed in split_info:
            train_mask, actual_seed, retry_offset = split_info[split_seed]
        else:
            train_mask, _, _, actual_seed, retry_offset = split_with_labels_record(state_ids, y, split_seed)
        scores = postmortem_score_vectors(data, train_mask, split_seed)
        alpha = safe_float(row.get("alpha"))
        weights = postmortem_weights(str(row.get("method")), float(alpha if alpha is not None else 1.0), scores, train_mask)
        wd = weight_diagnostics(weights, y[train_mask])
        out: Dict[str, object] = {
            "data_seed": data_seed,
            "split_seed": split_seed,
            "actual_split_seed": int(actual_seed),
            "split_retry_offset": int(retry_offset),
            "method": row.get("method"),
            "selected_alpha": alpha,
            "test_positive_rate": safe_float(row.get("test_positive_rate")),
        }
        out.update(wd)
        for metric in ("auprc", "auroc", "precision", "recall", "f1", "balanced_acc", "threshold"):
            value = safe_float(row.get(f"test_{metric}"))
            out[f"test_{metric}"] = value
            if uniform is not None and metric != "threshold":
                u = safe_float(uniform.get(f"test_{metric}"))
                out[f"test_{metric}_delta_vs_uniform"] = None if value is None or u is None else value - u
        rows.append(out)
    return rows


def find_dataset_dirs(root: Path) -> List[Tuple[int, Path]]:
    dirs: List[Tuple[int, Path]] = []
    for seed_dir in sorted(root.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        try:
            data_seed = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        data_path = seed_dir / "data" / "task_dataset.npz"
        if data_path.exists():
            dirs.append((data_seed, seed_dir))
    return dirs


def mean_of(rows: Sequence[Mapping[str, object]], key: str) -> float | None:
    vals = [safe_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    out = Path(args.out)
    ensure_dir(out)
    dataset_dirs = find_dataset_dirs(root)
    if not dataset_dirs:
        raise FileNotFoundError(f"No seed_*/data/task_dataset.npz datasets found under {root}")

    uniform_rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []
    oracle_feature_rows: List[Dict[str, object]] = []
    weighting_rows: List[Dict[str, object]] = []

    for data_seed, seed_dir in dataset_dirs:
        data_npz = np.load(seed_dir / "data" / "task_dataset.npz", allow_pickle=True)
        data: Dict[str, np.ndarray] = {k: data_npz[k] for k in data_npz.files}
        y = require(data, "target_task_success").astype(np.float32).reshape(-1)
        state_ids = require(data, "state_id").astype(np.int64)
        feature_sets = build_feature_sets(data, args.channel_dim, args.seed + data_seed * 10000)
        for split_seed in args.split_seeds:
            train_mask, val_mask, test_mask, actual_seed, retry_offset = split_with_labels_record(state_ids, y, int(split_seed))
            prevalence = float(np.mean(y[test_mask] > 0.5))
            constant_ap = average_precision(y[test_mask], np.zeros(int(np.sum(test_mask)), dtype=np.float32))
            for name, X in feature_sets.items():
                metrics = train_eval_feature_set(X, y, train_mask, val_mask, test_mask, args.epochs, args.lr, args.l2)
                row: Dict[str, object] = {
                    "data_seed": data_seed,
                    "split_seed": int(split_seed),
                    "actual_split_seed": int(actual_seed),
                    "split_retry_offset": int(retry_offset),
                    "feature_set": name,
                    "n_train": int(np.sum(train_mask)),
                    "n_val": int(np.sum(val_mask)),
                    "n_test": int(np.sum(test_mask)),
                    "train_positive_rate": float(np.mean(y[train_mask] > 0.5)),
                    "val_positive_rate": float(np.mean(y[val_mask] > 0.5)),
                    "test_positive_rate": prevalence,
                    "test_prevalence": prevalence,
                    "constant_score_auprc": constant_ap,
                    "test_auprc_lift_over_prevalence": metrics["test_auprc"] - prevalence,
                    "test_auprc_lift_over_constant": metrics["test_auprc"] - constant_ap,
                }
                row.update(metrics)
                feature_rows.append(row)
                if name == "obs_rgb_action":
                    uniform_rows.append(row)

        oracle_feature_rows.extend(
            oracle_as_feature_rows(
                feature_sets["obs_rgb_action"],
                y,
                state_ids,
                args.split_seeds,
                data_seed,
                args.epochs,
                args.lr,
                args.l2,
            )
        )
        weighting_rows.extend(
            selected_weighting_breakdown(seed_dir / "task_oracle" / "reports", data_seed, data, args.split_seeds)
        )

    write_csv(out / "uniform_absolute_metrics.csv", uniform_rows)
    write_csv(out / "action_only_vs_obs_only.csv", feature_rows)
    write_csv(out / "oracle_as_feature_upper_bound.csv", oracle_feature_rows)
    write_csv(out / "weighting_metric_breakdown.csv", weighting_rows)

    oracle_feature_mean = mean_of(oracle_feature_rows, "test_auprc")
    oracle_feature_min_vals = [safe_float(r.get("test_auprc")) for r in oracle_feature_rows]
    oracle_feature_min_vals = [v for v in oracle_feature_min_vals if v is not None]
    oracle_feature_min = float(np.min(oracle_feature_min_vals)) if oracle_feature_min_vals else None
    uniform_mean_auprc = mean_of(uniform_rows, "test_auprc")
    uniform_mean_prevalence = mean_of(uniform_rows, "test_prevalence")
    uniform_mean_lift = mean_of(uniform_rows, "test_auprc_lift_over_prevalence")

    feature_summary = []
    for name in sorted({str(r["feature_set"]) for r in feature_rows}):
        rows = [r for r in feature_rows if r["feature_set"] == name]
        feature_summary.append({
            "feature_set": name,
            "mean_test_auprc": mean_of(rows, "test_auprc"),
            "mean_test_auroc": mean_of(rows, "test_auroc"),
            "mean_test_f1": mean_of(rows, "test_f1"),
            "mean_test_recall": mean_of(rows, "test_recall"),
            "mean_test_precision": mean_of(rows, "test_precision"),
            "mean_test_auprc_lift_over_prevalence": mean_of(rows, "test_auprc_lift_over_prevalence"),
        })
    write_csv(out / "feature_set_summary.csv", feature_summary)

    oracle_sanity_pass = bool(
        oracle_feature_mean is not None
        and oracle_feature_min is not None
        and oracle_feature_mean >= args.oracle_feature_mean_pass
        and oracle_feature_min >= args.oracle_feature_min_pass
    )
    task_learnable = bool(uniform_mean_lift is not None and uniform_mean_lift >= args.learnable_lift)
    if not oracle_sanity_pass:
        branch = "case_c_oracle_feature_or_metric_bug"
    elif task_learnable:
        branch = "case_a_task_learnable_weighting_gate_inappropriate"
    else:
        branch = "case_b_task_or_features_weak"

    decision = {
        "branch": branch,
        "n_datasets": len(dataset_dirs),
        "split_seeds": [int(s) for s in args.split_seeds],
        "oracle_as_feature_mean_auprc": oracle_feature_mean,
        "oracle_as_feature_min_auprc": oracle_feature_min,
        "oracle_feature_sanity_pass": oracle_sanity_pass,
        "oracle_feature_mean_pass_threshold": float(args.oracle_feature_mean_pass),
        "oracle_feature_min_pass_threshold": float(args.oracle_feature_min_pass),
        "uniform_mean_test_auprc": uniform_mean_auprc,
        "uniform_mean_test_prevalence": uniform_mean_prevalence,
        "uniform_mean_auprc_lift_over_prevalence": uniform_mean_lift,
        "task_learnable_by_uniform_lift": task_learnable,
        "learnable_lift_threshold": float(args.learnable_lift),
        "interpretation": (
            "Task is learnable and oracle-as-feature sanity passes; the failed Step-2 sample-weighting gate "
            "should not be treated as an oracle upper-bound failure."
            if branch == "case_a_task_learnable_weighting_gate_inappropriate"
            else "See branch for postmortem interpretation."
        ),
    }
    write_json(out / "actioniv_gate_postmortem_decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root containing seed_*/data/task_dataset.npz and task_oracle reports.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--channel-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--learnable-lift", type=float, default=0.10)
    parser.add_argument("--oracle-feature-mean-pass", type=float, default=0.99)
    parser.add_argument("--oracle-feature-min-pass", type=float, default=0.98)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
