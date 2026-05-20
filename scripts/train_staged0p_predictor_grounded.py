#!/usr/bin/env python
"""Stage D0': predictor-grounded observability smoke.

This script is intentionally NumPy-only and lightweight. It tests exactly one
predictor-grounded observability redefinition after Stage C0/C0.5 No-go.

Primary score:
    predictor_grounded = rank(entropy(uniform_event_predictor)) * rank(obs_geom)

The script is not PSP/Dreamer and is not a detector-tweak loop. It implements the
precommitted D0' branch test described in docs/staged0p_precommit.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

STAGEB_REFERENCE_SIGNAL = 0.05
DIAGNOSTIC_ONLY_CHANNELS = {"event_response", "semantic", "edge"}
DEFAULT_INPUT_CHANNELS = ("rgb",)
DEFAULT_OBS_CHANNELS = ("rgb", "range")
DEFAULT_METHODS = (
    "uniform",
    "change_mask",
    "observability",
    "shuffled_observability",
    "predictor_uncertainty",
    "predictor_grounded",
    "shuffled_predictor_grounded",
    "lossgrad_observability",
    "oracle_event",
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def require_key(data: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing required key {key!r}. Available keys include {sorted(data.keys())[:40]}")
    return np.asarray(data[key])


def select_dense(data: Mapping[str, np.ndarray], key: str, dense_indices: np.ndarray) -> np.ndarray:
    arr = require_key(data, key)
    n_dense = dense_indices.size
    if arr.shape[0] == n_dense:
        return np.asarray(arr)
    max_idx = int(dense_indices.max()) if dense_indices.size else -1
    if arr.shape[0] > max_idx:
        return np.asarray(arr[dense_indices])
    raise ValueError(
        f"Array {key!r} has first dimension {arr.shape[0]}, cannot match dense length {n_dense} "
        f"or full max index {max_idx}."
    )


def flatten2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(x.shape[0], -1)


def average_rank(values: np.ndarray) -> np.ndarray:
    """Tie-aware [0,1] average rank."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size <= 1:
        return np.zeros_like(x, dtype=np.float32)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty_like(x, dtype=np.float64)
    start = 0
    while start < x.size:
        end = start + 1
        while end < x.size and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * float(start + end - 1)
        start = end
    return (ranks / float(x.size - 1)).astype(np.float32)


def rank_train_apply(train_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Approximate train-fitted percentile rank applied to arbitrary values.

    This avoids using eval/test labels to define score ranks. It is not exact
    average-rank for eval ties, but it is stable and train-only.
    """
    train = np.asarray(train_values, dtype=np.float64).reshape(-1)
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if train.size <= 1:
        return np.zeros_like(vals, dtype=np.float32)
    train_sorted = np.sort(train, kind="mergesort")
    left = np.searchsorted(train_sorted, vals, side="left")
    right = np.searchsorted(train_sorted, vals, side="right")
    mid = 0.5 * (left + right - 1)
    return np.clip(mid / float(train.size - 1), 0.0, 1.0).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))).astype(np.float32)


def standardize_train_apply(x_train: np.ndarray, x_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x_train - mean) / std).astype(np.float32), ((x_all - mean) / std).astype(np.float32), mean, std


def random_project(arr: np.ndarray, out_dim: int, seed: int) -> Tuple[np.ndarray, np.ndarray | None]:
    x = flatten2(arr)
    if out_dim <= 0 or out_dim >= x.shape[1]:
        return x.astype(np.float32), None
    rng = np.random.default_rng(seed)
    proj = rng.normal(0.0, 1.0 / math.sqrt(float(out_dim)), size=(x.shape[1], out_dim)).astype(np.float32)
    return (x @ proj).astype(np.float32), proj


def one_hot_strings(values: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    values = np.asarray(values).astype(str)
    out = np.zeros((values.size, len(classes)), dtype=np.float32)
    idx = {c: i for i, c in enumerate(classes)}
    for i, v in enumerate(values):
        if v in idx:
            out[i, idx[v]] = 1.0
    return out


def split_by_state(state_ids: np.ndarray, seed: int, train_frac: float, val_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = np.unique(state_ids.astype(np.int64))
    rng.shuffle(unique)
    n = unique.size
    n_train = max(1, int(round(train_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_set = set(unique[:n_train].tolist())
    val_set = set(unique[n_train : n_train + n_val].tolist())
    test_set = set(unique[n_train + n_val :].tolist())
    train = np.array([int(s) in train_set for s in state_ids], dtype=bool)
    val = np.array([int(s) in val_set for s in state_ids], dtype=bool)
    test = np.array([int(s) in test_set for s in state_ids], dtype=bool)
    return train, val, test


def binary_targets_from_event(delta_event: np.ndarray) -> np.ndarray:
    de = np.asarray(delta_event, dtype=np.float32)
    if de.ndim == 1:
        mag = np.abs(de)
    else:
        mag = np.linalg.norm(de.reshape(de.shape[0], -1), axis=1)
    return (mag > 1e-6).astype(np.float32)


def changed_targets_from_world_delta(world_delta: np.ndarray, q: float = 0.10) -> np.ndarray:
    wd = np.asarray(world_delta, dtype=np.float32)
    mag = np.linalg.norm(wd.reshape(wd.shape[0], -1), axis=1)
    threshold = float(np.quantile(mag, q))
    return (mag > max(threshold, 1e-8)).astype(np.float32)


def precision_recall_f1(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    y = np.asarray(y_true).astype(bool).reshape(-1)
    pred = np.asarray(y_score).reshape(-1) >= float(threshold)
    tp = float(np.sum(y & pred))
    fp = float(np.sum(~y & pred))
    fn = float(np.sum(y & ~pred))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    return float(precision), float(recall), float(f1)


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    scores = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return 0.5
    thresholds = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    best_t = float(thresholds[0]) if thresholds.size else 0.5
    best_f1 = -1.0
    for t in thresholds:
        _, _, f1 = precision_recall_f1(y_true, scores, float(t))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(np.int32).reshape(-1)
    score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_y = y[order]
    tp_seen = 0
    total_seen = 0
    ap = 0.0
    start = 0
    while start < sorted_y.size:
        end = start + 1
        while end < sorted_y.size and sorted_score[end] == sorted_score[start]:
            end += 1
        group_pos = int(sorted_y[start:end].sum())
        tp_seen += group_pos
        total_seen = end
        if group_pos:
            precision_at_group_end = tp_seen / max(total_seen, 1)
            ap += precision_at_group_end * group_pos / pos
        start = end
    return float(ap)


def auroc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(np.int32).reshape(-1)
    score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_rank(score) * max(1, score.size - 1) + 1.0
    sum_pos = float(np.sum(ranks[y == 1]))
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def balanced_accuracy(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    y = np.asarray(y_true).astype(bool).reshape(-1)
    pred = np.asarray(y_score).reshape(-1) >= float(threshold)
    tp = float(np.sum(y & pred))
    tn = float(np.sum(~y & ~pred))
    fp = float(np.sum(~y & pred))
    fn = float(np.sum(y & ~pred))
    tpr = tp / (tp + fn + 1e-12)
    tnr = tn / (tn + fp + 1e-12)
    return float(0.5 * (tpr + tnr))


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    epochs: int,
    lr: float,
    l2: float,
) -> Tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    w = np.maximum(w, 1e-6)
    w = w / (np.mean(w) + 1e-12)
    theta = np.zeros(X.shape[1], dtype=np.float64)
    bias = 0.0
    denom = float(w.sum()) + 1e-12
    for _ in range(int(epochs)):
        logits = X @ theta + bias
        p = sigmoid(logits).astype(np.float64)
        err = (p - y) * w
        grad = (X.T @ err) / denom + float(l2) * theta
        grad_b = float(np.sum(err) / denom)
        theta -= float(lr) * grad
        bias -= float(lr) * grad_b
    return theta.astype(np.float32), float(bias)


def predict_logistic(X: np.ndarray, theta: np.ndarray, bias: float) -> np.ndarray:
    return sigmoid(np.asarray(X, dtype=np.float32) @ theta.astype(np.float32) + float(bias))


def finite_max(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return max(finite) if finite else float("nan")


def crossfit_train_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    epochs: int,
    lr: float,
    l2: float,
    n_folds: int = 3,
) -> np.ndarray:
    """Out-of-fold train predictions for predictor-grounded weights."""
    y = np.asarray(y_train, dtype=np.float32).reshape(-1)
    preds = np.full(y.size, float(np.mean(y)), dtype=np.float32)
    if y.size < max(4, n_folds) or np.unique(y).size < 2:
        return preds
    rng = np.random.default_rng(int(seed) + 771_331)
    order = rng.permutation(y.size)
    folds = np.array_split(order, min(n_folds, y.size))
    for fold in folds:
        if fold.size == 0:
            continue
        train_idx = np.setdiff1d(order, fold, assume_unique=False)
        if train_idx.size == 0 or np.unique(y[train_idx]).size < 2:
            continue
        theta, bias = fit_logistic(
            X_train[train_idx],
            y[train_idx],
            np.ones(train_idx.size, dtype=np.float32),
            epochs=epochs,
            lr=lr,
            l2=l2,
        )
        preds[fold] = predict_logistic(X_train[fold], theta, bias)
    return preds


def metric_row(prefix: str, y: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    precision, recall, f1 = precision_recall_f1(y, score, threshold)
    return {
        f"{prefix}_auprc": average_precision(y, score),
        f"{prefix}_auroc": auroc_score(y, score),
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_balanced_acc": balanced_accuracy(y, score, threshold),
        f"{prefix}_positive_rate": float(np.mean(y)),
        f"{prefix}_threshold": float(threshold),
    }


def weight_diagnostics(weights: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    y = np.asarray(y_train).astype(bool).reshape(-1)
    w = np.maximum(w, 1e-12)
    ess = float((w.sum() ** 2) / np.sum(w ** 2))
    order = np.argsort(-w)
    n_top = max(1, int(math.ceil(0.10 * w.size)))
    top_mass = float(w[order[:n_top]].sum() / w.sum())
    pos_mass = float(w[y].sum() / w.sum()) if np.any(y) else 0.0
    neg_mass = float(w[~y].sum() / w.sum()) if np.any(~y) else 0.0
    return {
        "weight_ess": ess,
        "weight_ess_frac": ess / float(w.size),
        "top_decile_weight_mass": top_mass,
        "positive_event_weight_mass": pos_mass,
        "negative_event_weight_mass": neg_mass,
        "mean_weight": float(w.mean()),
        "max_weight": float(w.max()),
    }


def build_features(
    data: Mapping[str, np.ndarray],
    dense_indices: np.ndarray,
    train_mask: np.ndarray,
    channels: Sequence[str],
    channel_dim: int,
    seed: int,
    action_array: np.ndarray,
    action_type: np.ndarray,
) -> np.ndarray:
    parts: List[np.ndarray] = []
    for ci, channel in enumerate(channels):
        if channel in DIAGNOSTIC_ONLY_CHANNELS:
            raise ValueError(f"Channel {channel!r} is diagnostic-only and cannot be used as predictor input.")
        arr = select_dense(data, f"obs0_{channel}", dense_indices).astype(np.float32)
        proj, _ = random_project(arr, channel_dim, seed + 1009 * ci)
        _, proj_all, _, _ = standardize_train_apply(proj[train_mask], proj)
        parts.append(proj_all.astype(np.float32))
    act = np.asarray(action_array, dtype=np.float32)
    _, act_all, _, _ = standardize_train_apply(act[train_mask], act)
    parts.append(act_all.astype(np.float32))
    classes = sorted(np.unique(action_type.astype(str)).tolist())
    parts.append(one_hot_strings(action_type, classes))
    return np.concatenate(parts, axis=1).astype(np.float32)


def compute_observability_scores(
    data: Mapping[str, np.ndarray],
    dense_indices: np.ndarray,
    train_mask: np.ndarray,
    obs_channels: Sequence[str],
) -> np.ndarray:
    ranks: List[np.ndarray] = []
    for ch in obs_channels:
        if ch in DIAGNOSTIC_ONLY_CHANNELS:
            raise ValueError(f"Channel {ch!r} is diagnostic-only and cannot be used for observability.")
        key = f"detect_{ch}"
        vals = select_dense(data, key, dense_indices).astype(np.float32).reshape(-1)
        ranks.append(rank_train_apply(vals[train_mask], vals))
    if not ranks:
        return np.zeros(dense_indices.size, dtype=np.float32)
    geom = ranks[0]
    for r in ranks[1:]:
        geom = geom * np.maximum(r, 1e-8)
    geom = np.power(np.maximum(geom, 0.0), 1.0 / float(len(ranks)))
    return geom.astype(np.float32)


def compute_change_score(data: Mapping[str, np.ndarray], dense_indices: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    if "world_delta" in data:
        wd = select_dense(data, "world_delta", dense_indices).astype(np.float32)
        mag = np.linalg.norm(wd.reshape(wd.shape[0], -1), axis=1)
    elif "delta_rgb" in data:
        dr = select_dense(data, "delta_rgb", dense_indices).astype(np.float32)
        mag = np.linalg.norm(dr.reshape(dr.shape[0], -1), axis=1)
    else:
        raise KeyError("Need world_delta or delta_rgb for change_mask score.")
    return rank_train_apply(mag[train_mask], mag)


def method_train_weights(
    method: str,
    alpha: float,
    *,
    y_train: np.ndarray,
    obs_train: np.ndarray,
    change_train: np.ndarray,
    entropy_train: np.ndarray,
    uncertainty_train: np.ndarray,
    predictor_grounded_train: np.ndarray,
    shuffled_obs_train: np.ndarray,
    shuffled_predictor_grounded_train: np.ndarray,
    lossgrad_train: np.ndarray,
) -> np.ndarray:
    n = y_train.size
    if method == "uniform":
        return np.ones(n, dtype=np.float32)
    if method == "oracle_event":
        return np.where(y_train > 0.5, float(alpha), 1.0).astype(np.float32)
    if method == "change_mask":
        score = change_train
    elif method == "observability":
        score = obs_train
    elif method == "shuffled_observability":
        score = shuffled_obs_train
    elif method == "predictor_uncertainty":
        score = uncertainty_train
    elif method == "predictor_grounded":
        score = predictor_grounded_train
    elif method == "shuffled_predictor_grounded":
        score = shuffled_predictor_grounded_train
    elif method == "lossgrad_observability":
        score = lossgrad_train * obs_train
    else:
        raise ValueError(f"Unknown method {method!r}")
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
    score = np.clip(score, 0.0, 1.0)
    return (1.0 + float(alpha) * score).astype(np.float32)


def run_one(
    data_path: Path,
    out_dir: Path,
    *,
    input_channels: Sequence[str],
    observability_channels: Sequence[str],
    methods: Sequence[str],
    alphas: Sequence[float],
    seeds: Sequence[int],
    channel_dim: int,
    epochs: int,
    lr: float,
    l2: float,
    train_frac: float,
    val_frac: float,
) -> None:
    data_npz = np.load(data_path, allow_pickle=True)
    data: Dict[str, np.ndarray] = {k: data_npz[k] for k in data_npz.files}
    dense_indices = require_key(data, "delta_sample_indices").astype(np.int64)
    state_ids_full = require_key(data, "state_id").astype(np.int64)
    action_array_full = require_key(data, "action_array").astype(np.float32)
    action_type_full = require_key(data, "action_type").astype(str)
    state_ids = state_ids_full[dense_indices]
    action_array = action_array_full[dense_indices]
    action_type = action_type_full[dense_indices]
    delta_event = select_dense(data, "delta_event_response", dense_indices).astype(np.float32)
    y = binary_targets_from_event(delta_event)

    rows: List[Dict[str, object]] = []
    metadata = {
        "data_path": str(data_path),
        "n_dense": int(dense_indices.size),
        "n_states": int(np.unique(state_ids).size),
        "input_channels": list(input_channels),
        "observability_channels": list(observability_channels),
        "methods": list(methods),
        "alphas": [float(a) for a in alphas],
        "seeds": [int(s) for s in seeds],
        "target_positive_rate": float(np.mean(y)),
        "stageb_reference_signal": STAGEB_REFERENCE_SIGNAL,
    }

    for seed in seeds:
        train_mask, val_mask, test_mask = split_by_state(state_ids, int(seed), train_frac, val_frac)
        X = build_features(data, dense_indices, train_mask, input_channels, channel_dim, int(seed), action_array, action_type)
        y_train = y[train_mask]
        y_val = y[val_mask]
        y_test = y[test_mask]
        obs_score = compute_observability_scores(data, dense_indices, train_mask, observability_channels)
        change_score = compute_change_score(data, dense_indices, train_mask)

        # Uniform base predictor used to define predictor-grounded scores.
        base_w = np.ones(int(train_mask.sum()), dtype=np.float32)
        theta0, bias0 = fit_logistic(X[train_mask], y_train, base_w, epochs=epochs, lr=lr, l2=l2)
        p_all = predict_logistic(X, theta0, bias0)
        p_train_for_weights = crossfit_train_predictions(
            X[train_mask],
            y_train,
            seed=int(seed),
            epochs=epochs,
            lr=lr,
            l2=l2,
        )
        p_for_weights = p_all.copy()
        p_for_weights[train_mask] = p_train_for_weights
        eps = 1e-8
        entropy = -(p_for_weights * np.log(p_for_weights + eps) + (1.0 - p_for_weights) * np.log(1.0 - p_for_weights + eps)) / math.log(2.0)
        uncertainty = 1.0 - np.abs(2.0 * p_for_weights - 1.0)
        lossgrad = np.abs(p_for_weights - y)
        entropy_rank = rank_train_apply(entropy[train_mask], entropy)
        uncertainty_rank = rank_train_apply(uncertainty[train_mask], uncertainty)
        lossgrad_rank = rank_train_apply(lossgrad[train_mask], lossgrad)
        obs_geom_rank = rank_train_apply(obs_score[train_mask], obs_score)
        predictor_grounded_score = entropy_rank * obs_geom_rank

        shuffle_rng = np.random.default_rng(int(seed) + 910_003)
        shuffled_obs = obs_geom_rank.copy()
        shuffle_rng.shuffle(shuffled_obs)
        shuffled_pg = predictor_grounded_score.copy()
        shuffle_rng.shuffle(shuffled_pg)

        for method in methods:
            method_alphas = [0.0] if method == "uniform" else list(alphas)
            for alpha in method_alphas:
                w_train = method_train_weights(
                    method,
                    float(alpha),
                    y_train=y_train,
                    obs_train=obs_score[train_mask],
                    change_train=change_score[train_mask],
                    entropy_train=entropy_rank[train_mask],
                    uncertainty_train=uncertainty_rank[train_mask],
                    predictor_grounded_train=predictor_grounded_score[train_mask],
                    shuffled_obs_train=shuffled_obs[train_mask],
                    shuffled_predictor_grounded_train=shuffled_pg[train_mask],
                    lossgrad_train=lossgrad_rank[train_mask],
                )
                theta, bias = fit_logistic(X[train_mask], y_train, w_train, epochs=epochs, lr=lr, l2=l2)
                p_val = predict_logistic(X[val_mask], theta, bias)
                p_test = predict_logistic(X[test_mask], theta, bias)
                threshold = choose_threshold(y_val, p_val)
                row: Dict[str, object] = {
                    "seed": int(seed),
                    "method": method,
                    "alpha": float(alpha),
                    "threshold": float(threshold),
                    "n_train": int(train_mask.sum()),
                    "n_val": int(val_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "diagnostic_only": bool(method == "lossgrad_observability"),
                    "uses_target_labels_for_weight": bool(method in {"oracle_event", "lossgrad_observability"}),
                }
                row.update(metric_row("val", y_val, p_val, threshold))
                row.update(metric_row("test", y_test, p_test, threshold))
                row.update(weight_diagnostics(w_train, y_train))
                rows.append(row)

    # Compute deltas against uniform per seed.
    uniform_by_seed: Dict[int, Dict[str, float]] = {}
    for r in rows:
        if r["method"] == "uniform":
            uniform_by_seed[int(r["seed"])] = {
                "test_auprc": float(r.get("test_auprc", np.nan)),
                "test_f1": float(r.get("test_f1", np.nan)),
                "test_auroc": float(r.get("test_auroc", np.nan)),
                "test_balanced_acc": float(r.get("test_balanced_acc", np.nan)),
            }
    for r in rows:
        base = uniform_by_seed.get(int(r["seed"]), {})
        for metric in ("test_auprc", "test_f1", "test_auroc", "test_balanced_acc"):
            if metric in base:
                r[f"{metric}_delta_vs_uniform"] = float(r.get(metric, np.nan)) - float(base[metric])
        r["test_auprc_delta_minus_stageb_ref"] = float(r.get("test_auprc_delta_vs_uniform", np.nan)) - STAGEB_REFERENCE_SIGNAL
        if STAGEB_REFERENCE_SIGNAL > 0:
            r["test_auprc_delta_over_stageb_ref"] = float(r.get("test_auprc_delta_vs_uniform", np.nan)) / STAGEB_REFERENCE_SIGNAL

    write_csv(out_dir / "reports" / "staged0p_results.csv", rows)
    write_json(out_dir / "reports" / "staged0p_metadata.json", metadata)

    # Compact decision summary.
    by_method_alpha: Dict[Tuple[str, float], List[Mapping[str, object]]] = {}
    for r in rows:
        by_method_alpha.setdefault((str(r["method"]), float(r["alpha"])), []).append(r)
    summary_rows: List[Dict[str, object]] = []
    for (method, alpha), group in sorted(by_method_alpha.items()):
        vals = lambda key: np.array([float(g.get(key, np.nan)) for g in group], dtype=np.float64)
        auprc_delta = vals("test_auprc_delta_vs_uniform")
        f1_delta = vals("test_f1_delta_vs_uniform")
        behavior_delta = np.maximum(auprc_delta, f1_delta)
        summary_rows.append(
            {
                "method": method,
                "alpha": alpha,
                "n": len(group),
                "test_auprc_mean": float(np.nanmean(vals("test_auprc"))),
                "test_auprc_delta_vs_uniform_mean": float(np.nanmean(auprc_delta)),
                "test_auprc_delta_vs_uniform_min": float(np.nanmin(auprc_delta)),
                "test_auprc_delta_vs_uniform_positive_count": int(np.sum(auprc_delta > 0)),
                "test_f1_delta_vs_uniform_mean": float(np.nanmean(f1_delta)),
                "test_f1_delta_vs_uniform_min": float(np.nanmin(f1_delta)),
                "test_f1_delta_vs_uniform_positive_count": int(np.sum(f1_delta > 0)),
                "test_behavior_delta_vs_uniform_mean": float(np.nanmean(behavior_delta)),
                "test_behavior_delta_vs_uniform_min": float(np.nanmin(behavior_delta)),
                "test_behavior_delta_vs_uniform_positive_count": int(np.sum(behavior_delta > 0)),
                "test_auroc_delta_vs_uniform_mean": float(np.nanmean(vals("test_auroc_delta_vs_uniform"))),
                "test_auprc_delta_over_stageb_ref_mean": float(np.nanmean(vals("test_auprc_delta_over_stageb_ref"))),
                "passes_plus_0p10_auprc": bool(np.nanmean(auprc_delta) >= 0.10),
                "passes_plus_0p05_auprc": bool(np.nanmean(auprc_delta) >= 0.05),
                "passes_plus_0p10_behavior": bool(np.nanmean(behavior_delta) >= 0.10),
                "passes_plus_0p05_behavior": bool(np.nanmean(behavior_delta) >= 0.05),
            }
        )
    write_csv(out_dir / "reports" / "staged0p_summary.csv", summary_rows)

    def best_delta(method: str, key: str = "test_auprc_delta_vs_uniform_mean") -> float:
        candidates = [r for r in summary_rows if r["method"] == method]
        if not candidates:
            return float("nan")
        return max(float(r[key]) for r in candidates)

    def best_behavior_delta(method: str) -> float:
        return best_delta(method, "test_behavior_delta_vs_uniform_mean")

    def best_row(method: str, key: str = "test_behavior_delta_vs_uniform_mean") -> Mapping[str, object]:
        candidates = [r for r in summary_rows if r["method"] == method]
        if not candidates:
            return {}
        return max(candidates, key=lambda r: float(r[key]))

    decision = {
        "oracle_event_best_auprc_delta": best_delta("oracle_event"),
        "oracle_event_best_f1_delta": best_delta("oracle_event", "test_f1_delta_vs_uniform_mean"),
        "observability_best_auprc_delta": best_delta("observability"),
        "observability_best_f1_delta": best_delta("observability", "test_f1_delta_vs_uniform_mean"),
        "predictor_grounded_best_auprc_delta": best_delta("predictor_grounded"),
        "predictor_grounded_best_f1_delta": best_delta("predictor_grounded", "test_f1_delta_vs_uniform_mean"),
        "predictor_uncertainty_best_auprc_delta": best_delta("predictor_uncertainty"),
        "lossgrad_observability_best_auprc_delta": best_delta("lossgrad_observability"),
        "shuffled_observability_best_auprc_delta": best_delta("shuffled_observability"),
        "shuffled_observability_best_f1_delta": best_delta("shuffled_observability", "test_f1_delta_vs_uniform_mean"),
        "shuffled_predictor_grounded_best_auprc_delta": best_delta("shuffled_predictor_grounded"),
        "shuffled_predictor_grounded_best_f1_delta": best_delta("shuffled_predictor_grounded", "test_f1_delta_vs_uniform_mean"),
        "change_mask_best_auprc_delta": best_delta("change_mask"),
        "change_mask_best_f1_delta": best_delta("change_mask", "test_f1_delta_vs_uniform_mean"),
        "oracle_event_best_behavior_delta": best_behavior_delta("oracle_event"),
        "observability_best_behavior_delta": best_behavior_delta("observability"),
        "predictor_grounded_best_behavior_delta": best_behavior_delta("predictor_grounded"),
        "predictor_uncertainty_best_behavior_delta": best_behavior_delta("predictor_uncertainty"),
        "lossgrad_observability_best_behavior_delta": best_behavior_delta("lossgrad_observability"),
        "shuffled_observability_best_behavior_delta": best_behavior_delta("shuffled_observability"),
        "shuffled_predictor_grounded_best_behavior_delta": best_behavior_delta("shuffled_predictor_grounded"),
        "change_mask_best_behavior_delta": best_behavior_delta("change_mask"),
        "oracle_event_best_behavior_row": best_row("oracle_event"),
        "predictor_grounded_best_auprc_row": best_row("predictor_grounded", "test_auprc_delta_vs_uniform_mean"),
        "predictor_grounded_best_f1_row": best_row("predictor_grounded", "test_f1_delta_vs_uniform_mean"),
        "predictor_grounded_best_behavior_row": best_row("predictor_grounded"),
    }
    oracle_auprc_pass = bool(decision["oracle_event_best_auprc_delta"] >= 0.10)
    oracle_f1_pass = bool(decision["oracle_event_best_f1_delta"] >= 0.10)
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
        and int(pg_auprc_best.get("test_auprc_delta_vs_uniform_positive_count", 0)) >= 4
    )
    pg_f1_pass_raw = bool(
        decision["predictor_grounded_best_f1_delta"] >= 0.10
        and decision["predictor_grounded_best_f1_delta"] > best_control_f1
        and int(pg_f1_best.get("test_f1_delta_vs_uniform_positive_count", 0)) >= 4
    )
    pg_pass_raw = bool(pg_auprc_pass_raw or pg_f1_pass_raw)
    pg_pass = bool(oracle_pass and pg_pass_raw)
    if oracle_pass and pg_pass:
        branch = "branch_1_best_case_proceed_to_c1"
    elif oracle_pass and not pg_pass:
        branch = "branch_2_stop_toy_or_environment_migration"
    else:
        branch = "branch_3_detector_failed_stop_toy"
    decision.update(
        {
            "oracle_pass_plus_0p10": oracle_pass,
            "oracle_auprc_pass_plus_0p10": oracle_auprc_pass,
            "oracle_f1_pass_plus_0p10": oracle_f1_pass,
            "predictor_grounded_raw_auprc_pass_plus_0p10_and_controls": pg_auprc_pass_raw,
            "predictor_grounded_raw_f1_pass_plus_0p10_and_controls": pg_f1_pass_raw,
            "predictor_grounded_raw_pass_plus_0p10_and_controls": pg_pass_raw,
            "predictor_grounded_pass_plus_0p10_and_controls": pg_pass,
            "best_control_behavior_delta": best_control,
            "best_control_auprc_delta": best_control_auprc,
            "best_control_f1_delta": best_control_f1,
            "predictor_grounded_control_margin": decision["predictor_grounded_best_behavior_delta"] - best_control,
            "predictor_grounded_auprc_control_margin": decision["predictor_grounded_best_auprc_delta"] - best_control_auprc,
            "predictor_grounded_f1_control_margin": decision["predictor_grounded_best_f1_delta"] - best_control_f1,
            "predictor_grounded_interpretable": bool(oracle_pass),
            "decision_branch": branch,
            "stageb_reference_signal": STAGEB_REFERENCE_SIGNAL,
        }
    )
    write_json(out_dir / "reports" / "staged0p_decision_summary.json", decision)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--input-channels", nargs="+", default=list(DEFAULT_INPUT_CHANNELS))
    p.add_argument("--observability-channels", nargs="+", default=list(DEFAULT_OBS_CHANNELS))
    p.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    p.add_argument("--alphas", nargs="+", type=float, default=[2.0, 4.0, 8.0, 16.0, 32.0])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--channel-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--train-frac", type=float, default=0.60)
    p.add_argument("--val-frac", type=float, default=0.20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)
    run_one(
        args.data,
        args.out,
        input_channels=args.input_channels,
        observability_channels=args.observability_channels,
        methods=args.methods,
        alphas=args.alphas,
        seeds=args.seeds,
        channel_dim=args.channel_dim,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )


if __name__ == "__main__":
    main()
