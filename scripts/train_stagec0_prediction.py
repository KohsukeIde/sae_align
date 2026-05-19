#!/usr/bin/env python
"""Stage C0 selective-prediction smoke test.

This script is intentionally lightweight and NumPy-only. It asks whether the
continuous action-effect observability signal that appeared in Stage B.6 has
behavioral utility when used as a sample-weighting signal for simple prediction
heads.

The script trains weighted ridge models on dense Stage-0 samples. It does not
attempt to reproduce PSP/Dreamer. Those are Stage C1+ baselines. Stage C0 is a
smoke test with a clear stop/go role:

  * If observability weighting cannot beat uniform/change-mask on event/OOD
    metrics, do not start PSP/Dreamer comparisons.
  * If it beats them with a larger effect size than the Stage-B kNN signal
    (~0.05), then Stage C1 is worth running.

Expected input: a Stage-0 dataset npz generated with --store-static-obs and
with dense deltas saved, e.g. keys such as obs0_rgb, delta_rgb, delta_range,
delta_event_response, world_delta, action_array, state_id, action_type,
delta_sample_indices.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


STAGEB_REFERENCE_SIGNAL = 0.05
DEFAULT_INPUT_CHANNELS = ("rgb",)
DEFAULT_OBSERVABILITY_CHANNELS = ("rgb", "range")
DIAGNOSTIC_ONLY_CHANNELS = ("event_response", "semantic", "edge")
DEFAULT_METHODS = (
    "uniform",
    "change_mask",
    "observability",
    "shuffled_observability",
    "oracle_event",
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def write_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_npz_array(data: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing required key {key!r}. Available sample: {sorted(list(data.keys()))[:30]}")
    return np.asarray(data[key])


def get_optional_array(data: Mapping[str, np.ndarray], key: str) -> np.ndarray | None:
    if key not in data:
        return None
    return np.asarray(data[key])


def flatten2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(x.shape[0], -1)


def percentile_rank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(x.size, dtype=np.float32)
    denom = max(1, x.size - 1)
    return ranks / float(denom)


def percentile_rank_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("percentile_rank_columns expects a 2-D array")
    return np.stack([percentile_rank(x[:, i]) for i in range(x.shape[1])], axis=1).astype(np.float32)


def standardize_train_apply(x_train: np.ndarray, x_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_train - mean) / std, (x_eval - mean) / std, mean, std


def weighted_ridge_fit(X: np.ndarray, Y: np.ndarray, weights: np.ndarray, ridge: float) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    w = np.maximum(w, 1e-6)
    sw = np.sqrt(w)[:, None]
    X1 = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
    Xw = X1 * sw
    Yw = Y * sw
    reg = float(ridge) * np.eye(X1.shape[1], dtype=np.float32)
    reg[-1, -1] = 0.0
    A = Xw.T @ Xw + reg
    B = Xw.T @ Yw
    coef = np.linalg.solve(A.astype(np.float64), B.astype(np.float64)).astype(np.float32)
    W = coef[:-1]
    b = coef[-1]
    return W, b


def linear_predict(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(X, dtype=np.float32) @ W + b).astype(np.float32)


def binary_f1(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(bool).reshape(-1)
    y_pred = (np.asarray(y_score).reshape(-1) >= float(threshold))
    tp = float(np.sum(y_true & y_pred))
    fp = float(np.sum(~y_true & y_pred))
    fn = float(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    return float(f1), float(precision), float(recall)


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray, *, n_grid: int = 51) -> float:
    y_score = np.asarray(y_score, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true).astype(bool).reshape(-1)
    if y_score.size == 0:
        return 0.0
    qs = np.linspace(0.01, 0.99, n_grid)
    thresholds = np.unique(np.quantile(y_score, qs))
    if thresholds.size == 0:
        return float(np.mean(y_score))
    best_t = float(thresholds[0])
    best_f1 = -1.0
    for t in thresholds:
        f1, _, _ = binary_f1(y_true, y_score, float(t))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def multi_binary_f1(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    return binary_f1(np.asarray(y_true).reshape(-1), np.asarray(y_score).reshape(-1), threshold)


def split_states(state_ids_dense: np.ndarray, seed: int, train_frac: float, val_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_states = np.unique(state_ids_dense.astype(np.int64))
    rng.shuffle(unique_states)
    n = unique_states.size
    n_train = max(1, int(round(train_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_states = set(unique_states[:n_train].tolist())
    val_states = set(unique_states[n_train : n_train + n_val].tolist())
    test_states = set(unique_states[n_train + n_val :].tolist())
    train = np.array([sid in train_states for sid in state_ids_dense], dtype=bool)
    val = np.array([sid in val_states for sid in state_ids_dense], dtype=bool)
    test = np.array([sid in test_states for sid in state_ids_dense], dtype=bool)
    return train, val, test


def random_project_fit_apply(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    out_dim: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_train = flatten2(x_train)
    x_eval = flatten2(x_eval)
    if out_dim <= 0 or out_dim >= x_train.shape[1]:
        xt, xe, _, _ = standardize_train_apply(x_train, x_eval)
        return xt.astype(np.float32), xe.astype(np.float32)
    rng = np.random.default_rng(seed)
    proj = rng.normal(0.0, 1.0 / math.sqrt(out_dim), size=(x_train.shape[1], out_dim)).astype(np.float32)
    xt = x_train @ proj
    xe = x_eval @ proj
    xt, xe, _, _ = standardize_train_apply(xt, xe)
    return xt.astype(np.float32), xe.astype(np.float32)


def one_hot_strings(values: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    values = np.asarray(values).astype(str)
    out = np.zeros((values.size, len(classes)), dtype=np.float32)
    index = {c: i for i, c in enumerate(classes)}
    for i, v in enumerate(values):
        if v in index:
            out[i, index[v]] = 1.0
    return out


@dataclass
class DatasetBundle:
    arrays: Mapping[str, np.ndarray]
    dense_indices: np.ndarray
    full_indices: np.ndarray
    state_ids: np.ndarray
    action_array: np.ndarray
    action_type: np.ndarray


def load_bundle(path: Path) -> DatasetBundle:
    data = np.load(path, allow_pickle=True)
    arrays: Dict[str, np.ndarray] = {k: data[k] for k in data.files}
    dense_indices = safe_npz_array(arrays, "delta_sample_indices").astype(np.int64)
    if dense_indices.ndim != 1:
        raise ValueError("delta_sample_indices must be 1-D")
    full_state_id = safe_npz_array(arrays, "state_id").astype(np.int64)
    full_action_array = safe_npz_array(arrays, "action_array").astype(np.float32)
    full_action_type = safe_npz_array(arrays, "action_type").astype(str)
    return DatasetBundle(
        arrays=arrays,
        dense_indices=dense_indices,
        full_indices=dense_indices,
        state_ids=full_state_id[dense_indices],
        action_array=full_action_array[dense_indices],
        action_type=full_action_type[dense_indices],
    )


def require_dense_static(bundle: DatasetBundle, channel: str) -> np.ndarray:
    key = f"obs0_{channel}"
    if key not in bundle.arrays:
        raise KeyError(
            f"Missing {key}. Stage C0 requires Stage-0 data generated with --store-static-obs. "
            f"Rerun make_stage0_dataset.py with --store-static-obs and dense samples."
        )
    return np.asarray(bundle.arrays[key], dtype=np.float32)


def require_dense_delta(bundle: DatasetBundle, channel: str) -> np.ndarray:
    key = f"delta_{channel}"
    if key not in bundle.arrays:
        raise KeyError(f"Missing {key}. Available dense delta keys: {[k for k in bundle.arrays if k.startswith('delta_')]}")
    return np.asarray(bundle.arrays[key], dtype=np.float32)


def build_features(
    bundle: DatasetBundle,
    input_channels: Sequence[str],
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    *,
    channel_dim: int,
    seed: int,
    ood: bool = False,
    ood_noise_std: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    train_parts: List[np.ndarray] = []
    eval_parts: List[np.ndarray] = []
    for ci, ch in enumerate(input_channels):
        obs = require_dense_static(bundle, ch)
        obs_train = obs[train_mask]
        obs_eval = obs[eval_mask].copy()
        if ood and ch == "rgb":
            rng = np.random.default_rng(seed + 1009)
            obs_eval = obs_eval + rng.normal(0.0, ood_noise_std, size=obs_eval.shape).astype(np.float32)
            obs_eval = np.clip(obs_eval, 0.0, 1.0)
        xt, xe = random_project_fit_apply(obs_train, obs_eval, out_dim=channel_dim, seed=seed + 97 * (ci + 1))
        train_parts.append(xt)
        eval_parts.append(xe)
    action_train = bundle.action_array[train_mask].astype(np.float32)
    action_eval = bundle.action_array[eval_mask].astype(np.float32)
    action_train, action_eval, _, _ = standardize_train_apply(action_train, action_eval)
    classes = ["place", "erase", "push", "noop"]
    type_train = one_hot_strings(bundle.action_type[train_mask], classes)
    type_eval = one_hot_strings(bundle.action_type[eval_mask], classes)
    train_parts.extend([action_train.astype(np.float32), type_train.astype(np.float32)])
    eval_parts.extend([action_eval.astype(np.float32), type_eval.astype(np.float32)])
    return np.concatenate(train_parts, axis=1), np.concatenate(eval_parts, axis=1)


def changed_cell_map(delta: np.ndarray, threshold: float) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float32)
    if delta.ndim == 4:
        mag = np.max(np.abs(delta), axis=1)
    elif delta.ndim == 3:
        mag = np.max(np.abs(delta), axis=1) if delta.shape[1] in (1, 3) else np.abs(delta)
    else:
        flat = flatten2(delta)
        return (np.abs(flat) > threshold).astype(np.float32)
    return (mag > threshold).astype(np.float32).reshape(delta.shape[0], -1)


def event_targets(
    bundle: DatasetBundle,
    event_channel: str,
    event_threshold: float,
    *,
    allow_world_delta_fallback: bool,
) -> np.ndarray:
    key = f"delta_{event_channel}"
    if key not in bundle.arrays:
        if not allow_world_delta_fallback:
            raise KeyError(
                f"Missing {key}. Stage C0 primary runs require an explicit event target. "
                "Use --allow-world-delta-event-fallback only for diagnostic fallback runs."
            )
        # Fall back to world-delta event-like scalar.
        wd = safe_npz_array(bundle.arrays, "world_delta").astype(np.float32)[bundle.dense_indices]
        return (wd[:, None] > np.quantile(wd, 0.75)).astype(np.float32)
    y = np.asarray(bundle.arrays[key], dtype=np.float32)
    if y.ndim > 2:
        y = y.reshape(y.shape[0], -1)
    return (np.abs(y) > float(event_threshold)).astype(np.float32)


def reconstruction_targets(bundle: DatasetBundle, target_channel: str) -> np.ndarray:
    return flatten2(require_dense_delta(bundle, target_channel)).astype(np.float32)


def detectability_matrix(
    bundle: DatasetBundle,
    channels: Sequence[str],
) -> np.ndarray:
    cols = []
    for ch in channels:
        key = f"detect_{ch}"
        if key not in bundle.arrays:
            raise KeyError(f"Missing {key}; cannot compute observability score.")
        dense_detect = np.asarray(bundle.arrays[key], dtype=np.float32)[bundle.dense_indices]
        cols.append(dense_detect)
    return np.stack(cols, axis=1).astype(np.float32)


def mix_observability_ranks(ranked: np.ndarray, *, mix: str) -> np.ndarray:
    ranked = np.asarray(ranked, dtype=np.float32)
    if mix == "mean":
        out = ranked.mean(axis=1)
    elif mix == "geom":
        out = np.sqrt(np.maximum(np.prod(ranked, axis=1), 0.0))
    elif mix == "product":
        out = np.prod(ranked, axis=1)
    elif mix == "min":
        out = np.min(ranked, axis=1)
    else:
        raise ValueError(f"Unknown observability mix {mix!r}")
    return out.astype(np.float32)


def normalize_weights(score: np.ndarray, alpha: float) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    score = percentile_rank(score)
    w = 1.0 + float(alpha) * score
    return (w / (np.mean(w) + 1e-12)).astype(np.float32)


def method_weights(
    method: str,
    *,
    observability_score: np.ndarray,
    changed_ratio: np.ndarray,
    event_present: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = observability_score.size
    if method == "uniform":
        return np.ones(n, dtype=np.float32)
    if method == "change_mask":
        return normalize_weights(changed_ratio, alpha)
    if method == "observability":
        return normalize_weights(observability_score, alpha)
    if method == "shuffled_observability":
        return normalize_weights(rng.permutation(observability_score), alpha)
    if method == "oracle_event":
        return normalize_weights(event_present.astype(np.float32), alpha)
    raise ValueError(f"Unknown method: {method}")


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    denom = float(np.sum(weights**2))
    if denom <= 0:
        return 0.0
    return float((np.sum(weights) ** 2) / denom)


def validate_bundle(bundle: DatasetBundle, *, input_channels: Sequence[str], observability_channels: Sequence[str]) -> None:
    blocked_inputs = sorted(set(input_channels).intersection(DIAGNOSTIC_ONLY_CHANNELS))
    if blocked_inputs:
        raise ValueError(
            f"Diagnostic/privileged channels cannot be C0 inputs: {blocked_inputs}. "
            "event_response is a target only, semantic is privileged, and edge is diagnostic-only."
        )
    blocked_obs = sorted(set(observability_channels).intersection(DIAGNOSTIC_ONLY_CHANNELS))
    if blocked_obs:
        raise ValueError(f"Diagnostic/privileged channels cannot define deployable C0 observability: {blocked_obs}")
    static_indices = get_optional_array(bundle.arrays, "static_obs_sample_indices")
    if static_indices is not None and not np.array_equal(static_indices.astype(np.int64), bundle.dense_indices):
        raise ValueError("static_obs_sample_indices must exactly match delta_sample_indices for Stage C0.")
    for ch in set(input_channels).union(observability_channels).union({str("rgb")}):
        key = f"obs0_{ch}"
        if ch in input_channels and key not in bundle.arrays:
            raise KeyError(f"Missing {key}; regenerate Stage-0 data with --store-static-obs.")


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def evaluate_method(
    *,
    method: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    X_test_ood: np.ndarray,
    Yrec_train: np.ndarray,
    Yrec_val: np.ndarray,
    Yrec_test: np.ndarray,
    Yevent_train: np.ndarray,
    Yevent_val: np.ndarray,
    Yevent_test: np.ndarray,
    Ychanged_train: np.ndarray,
    Ychanged_val: np.ndarray,
    Ychanged_test: np.ndarray,
    weights_train: np.ndarray,
    ridge: float,
) -> Dict[str, float]:
    W_rec, b_rec = weighted_ridge_fit(X_train, Yrec_train, weights_train, ridge)
    pred_rec_val = linear_predict(X_val, W_rec, b_rec)
    pred_rec_test = linear_predict(X_test, W_rec, b_rec)
    pred_rec_ood = linear_predict(X_test_ood, W_rec, b_rec)

    W_event, b_event = weighted_ridge_fit(X_train, Yevent_train, weights_train, ridge)
    pred_event_val = linear_predict(X_val, W_event, b_event)
    pred_event_test = linear_predict(X_test, W_event, b_event)
    pred_event_ood = linear_predict(X_test_ood, W_event, b_event)

    rec_mse = mse(Yrec_test, pred_rec_test)
    rec_mse_ood = mse(Yrec_test, pred_rec_ood)

    # Targets are change/event occurrence indicators built from absolute deltas.
    # Score with predicted magnitudes so negative predicted deltas are not
    # incorrectly treated as low-confidence no-change events.
    pred_event_val_score = np.abs(pred_event_val)
    pred_event_test_score = np.abs(pred_event_test)
    pred_event_ood_score = np.abs(pred_event_ood)
    pred_rec_val_score = np.abs(pred_rec_val)
    pred_rec_test_score = np.abs(pred_rec_test)
    pred_rec_ood_score = np.abs(pred_rec_ood)

    event_threshold = choose_threshold(Yevent_val, pred_event_val_score)
    event_f1, event_prec, event_rec = multi_binary_f1(Yevent_test, pred_event_test_score, event_threshold)
    event_f1_ood, event_prec_ood, event_rec_ood = multi_binary_f1(Yevent_test, pred_event_ood_score, event_threshold)

    changed_threshold = choose_threshold(Ychanged_val, pred_rec_val_score)
    changed_f1, changed_prec, changed_rec = multi_binary_f1(Ychanged_test, pred_rec_test_score, changed_threshold)
    changed_f1_ood, _, _ = multi_binary_f1(Ychanged_test, pred_rec_ood_score, changed_threshold)

    return {
        "method": method,
        "reconstruction_mse": rec_mse,
        "reconstruction_mse_ood": rec_mse_ood,
        "event_f1": event_f1,
        "event_precision": event_prec,
        "event_recall": event_rec,
        "event_f1_ood": event_f1_ood,
        "event_precision_ood": event_prec_ood,
        "event_recall_ood": event_rec_ood,
        "changed_cell_f1": changed_f1,
        "changed_cell_precision": changed_prec,
        "changed_cell_recall": changed_rec,
        "changed_cell_f1_ood": changed_f1_ood,
        "event_threshold": float(event_threshold),
        "changed_threshold": float(changed_threshold),
    }


def summarize(rows: List[Mapping[str, object]]) -> List[Dict[str, object]]:
    methods = sorted({str(r["method"]) for r in rows})
    metrics = [
        "reconstruction_mse",
        "reconstruction_mse_ood",
        "event_f1",
        "event_f1_ood",
        "changed_cell_f1",
        "changed_cell_f1_ood",
    ]
    out: List[Dict[str, object]] = []
    uniform_means: Dict[str, float] = {}
    for metric in metrics:
        vals = [float(r[metric]) for r in rows if str(r["method"]) == "uniform"]
        uniform_means[metric] = float(np.mean(vals)) if vals else float("nan")
    for method in methods:
        subset = [r for r in rows if str(r["method"]) == method]
        row: Dict[str, object] = {"method": method, "n_runs": len(subset)}
        for metric in metrics:
            vals = np.array([float(r[metric]) for r in subset], dtype=np.float32)
            mean = float(np.mean(vals)) if vals.size else float("nan")
            std = float(np.std(vals)) if vals.size else float("nan")
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            if metric in uniform_means and np.isfinite(uniform_means[metric]):
                # For MSE, lower is better, so positive improvement means uniform - method.
                if "mse" in metric:
                    delta = uniform_means[metric] - mean
                else:
                    delta = mean - uniform_means[metric]
                row[f"{metric}_delta_vs_uniform"] = float(delta)
                row[f"{metric}_delta_minus_stageb_ref"] = float(delta - STAGEB_REFERENCE_SIGNAL)
                row[f"{metric}_delta_over_stageb_ref"] = float(delta / STAGEB_REFERENCE_SIGNAL)
        out.append(row)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage C0 minimal selective-prediction smoke test.")
    p.add_argument("--data", type=str, required=True, help="Stage-0 dataset npz with static obs and dense deltas.")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--input-channels", nargs="*", default=list(DEFAULT_INPUT_CHANNELS))
    p.add_argument("--target-channel", type=str, default="rgb")
    p.add_argument("--event-channel", type=str, default="event_response")
    p.add_argument("--observability-channels", nargs="*", default=list(DEFAULT_OBSERVABILITY_CHANNELS))
    p.add_argument("--observability-mix", choices=["geom", "mean", "product", "min"], default="geom")
    p.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--train-frac", type=float, default=0.60)
    p.add_argument("--val-frac", type=float, default=0.20)
    p.add_argument("--channel-dim", type=int, default=128)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--weight-alpha", type=float, default=4.0)
    p.add_argument("--changed-threshold", type=float, default=1e-4)
    p.add_argument("--event-threshold", type=float, default=1e-4)
    p.add_argument("--ood-noise-std", type=float, default=0.35)
    p.add_argument("--max-samples", type=int, default=0, help="Optional cap on dense samples for smoke tests.")
    p.add_argument(
        "--allow-world-delta-event-fallback",
        action="store_true",
        help="Diagnostic-only fallback when delta_event_response is unavailable.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    reports = out / "reports"
    ensure_dir(reports)

    bundle = load_bundle(Path(args.data))
    validate_bundle(bundle, input_channels=args.input_channels, observability_channels=args.observability_channels)
    n_dense = bundle.dense_indices.size
    selection = np.arange(n_dense)
    if int(args.max_samples) > 0 and int(args.max_samples) < n_dense:
        rng = np.random.default_rng(123)
        selection = np.sort(rng.choice(selection, size=int(args.max_samples), replace=False))
        # Create a shallow filtered bundle.
        arrays = dict(bundle.arrays)
        for key in list(arrays.keys()):
            if key.startswith("delta_") or key.startswith("obs0_"):
                if arrays[key].shape[0] == n_dense:
                    arrays[key] = arrays[key][selection]
        if "static_obs_sample_indices" in arrays and arrays["static_obs_sample_indices"].shape[0] == n_dense:
            arrays["static_obs_sample_indices"] = arrays["static_obs_sample_indices"][selection]
        if "delta_sample_indices" in arrays and arrays["delta_sample_indices"].shape[0] == n_dense:
            arrays["delta_sample_indices"] = arrays["delta_sample_indices"][selection]
        bundle = DatasetBundle(
            arrays=arrays,
            dense_indices=bundle.dense_indices[selection],
            full_indices=bundle.full_indices[selection],
            state_ids=bundle.state_ids[selection],
            action_array=bundle.action_array[selection],
            action_type=bundle.action_type[selection],
        )

    Yrec = reconstruction_targets(bundle, args.target_channel)
    # Use the same flattened target dimensionality as reconstruction for C0.
    # A later neural C1 can replace this with spatial cell-level metrics.
    Ychanged = (np.abs(Yrec) > float(args.changed_threshold)).astype(np.float32)
    Yevent = event_targets(
        bundle,
        args.event_channel,
        args.event_threshold,
        allow_world_delta_fallback=bool(args.allow_world_delta_event_fallback),
    )

    detect_matrix = detectability_matrix(bundle, args.observability_channels)
    changed_ratio = np.mean(Ychanged, axis=1).astype(np.float32)
    event_present = np.any(Yevent > 0.5, axis=1).astype(np.float32)

    rows: List[Dict[str, object]] = []
    split_rows: List[Dict[str, object]] = []
    for seed in args.seeds:
        train_mask, val_mask, test_mask = split_states(bundle.state_ids, seed, args.train_frac, args.val_frac)
        split_rows.append(
            {
                "seed": seed,
                "n_train": int(np.sum(train_mask)),
                "n_val": int(np.sum(val_mask)),
                "n_test": int(np.sum(test_mask)),
                "n_train_states": int(np.unique(bundle.state_ids[train_mask]).size),
                "n_val_states": int(np.unique(bundle.state_ids[val_mask]).size),
                "n_test_states": int(np.unique(bundle.state_ids[test_mask]).size),
            }
        )
        X_train, X_val = build_features(
            bundle,
            args.input_channels,
            train_mask,
            val_mask,
            channel_dim=args.channel_dim,
            seed=seed,
            ood=False,
        )
        _, X_test = build_features(
            bundle,
            args.input_channels,
            train_mask,
            test_mask,
            channel_dim=args.channel_dim,
            seed=seed,
            ood=False,
        )
        _, X_test_ood = build_features(
            bundle,
            args.input_channels,
            train_mask,
            test_mask,
            channel_dim=args.channel_dim,
            seed=seed,
            ood=True,
            ood_noise_std=args.ood_noise_std,
        )
        rng = np.random.default_rng(seed + 2027)
        obs_train_ranks = percentile_rank_columns(detect_matrix[train_mask])
        observability_score_train = mix_observability_ranks(obs_train_ranks, mix=args.observability_mix)
        for method in args.methods:
            weights_train = method_weights(
                method,
                observability_score=observability_score_train,
                changed_ratio=changed_ratio[train_mask],
                event_present=event_present[train_mask],
                alpha=args.weight_alpha,
                rng=rng,
            )
            result = evaluate_method(
                method=method,
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                X_test_ood=X_test_ood,
                Yrec_train=Yrec[train_mask],
                Yrec_val=Yrec[val_mask],
                Yrec_test=Yrec[test_mask],
                Yevent_train=Yevent[train_mask],
                Yevent_val=Yevent[val_mask],
                Yevent_test=Yevent[test_mask],
                Ychanged_train=Ychanged[train_mask],
                Ychanged_val=Ychanged[val_mask],
                Ychanged_test=Ychanged[test_mask],
                weights_train=weights_train,
                ridge=args.ridge,
            )
            result.update(
                {
                    "seed": seed,
                    "n_train": int(np.sum(train_mask)),
                    "n_val": int(np.sum(val_mask)),
                    "n_test": int(np.sum(test_mask)),
                    "weight_mean_train": float(np.mean(weights_train)),
                    "weight_std_train": float(np.std(weights_train)),
                    "weight_ess_train": float(effective_sample_size(weights_train)),
                    "weight_top_decile_mass_train": float(
                        np.sum(np.sort(weights_train)[-max(1, weights_train.size // 10) :])
                        / (np.sum(weights_train) + 1e-12)
                    ),
                    "stageb_reference_signal": STAGEB_REFERENCE_SIGNAL,
                }
            )
            rows.append(result)

    summary = summarize(rows)
    write_csv(reports / "stagec0_results.csv", rows)
    write_csv(reports / "stagec0_summary.csv", summary)
    write_csv(reports / "stagec0_splits.csv", split_rows)

    # Small decision summary with explicit effect-size comparison to the B.6 kNN signal.
    by_method = {str(r["method"]): r for r in summary}
    obs = by_method.get("observability", {})
    chg = by_method.get("change_mask", {})
    uniform = by_method.get("uniform", {})
    decision = {
        "stage": "C0",
        "data": str(args.data),
        "input_channels": list(args.input_channels),
        "target_channel": args.target_channel,
        "event_channel": args.event_channel,
        "observability_channels": list(args.observability_channels),
        "observability_mix": args.observability_mix,
        "stageb_reference_signal": STAGEB_REFERENCE_SIGNAL,
        "observability_event_f1_delta_vs_uniform": obs.get("event_f1_delta_vs_uniform"),
        "observability_event_f1_delta_minus_stageb_ref": obs.get("event_f1_delta_minus_stageb_ref"),
        "observability_ood_event_f1_delta_vs_uniform": obs.get("event_f1_ood_delta_vs_uniform"),
        "observability_changed_f1_delta_vs_uniform": obs.get("changed_cell_f1_delta_vs_uniform"),
        "observability_rec_mse_improvement_vs_uniform": obs.get("reconstruction_mse_delta_vs_uniform"),
        "change_mask_event_f1_delta_vs_uniform": chg.get("event_f1_delta_vs_uniform"),
        "uniform_reconstruction_mse_mean": uniform.get("reconstruction_mse_mean"),
        "go_condition_observability_beats_uniform_event": bool(
            (obs.get("event_f1_delta_vs_uniform") is not None)
            and float(obs.get("event_f1_delta_vs_uniform")) > 0.0
        ),
        "go_condition_effect_exceeds_stageb_ref_event": bool(
            (obs.get("event_f1_delta_vs_uniform") is not None)
            and float(obs.get("event_f1_delta_vs_uniform")) > STAGEB_REFERENCE_SIGNAL
        ),
        "go_condition_observability_beats_change_event": bool(
            (obs.get("event_f1_mean") is not None)
            and (chg.get("event_f1_mean") is not None)
            and float(obs.get("event_f1_mean")) > float(chg.get("event_f1_mean"))
        ),
    }
    write_json(reports / "stagec0_decision_summary.json", decision)
    print(f"Wrote Stage C0 reports to {reports}")


if __name__ == "__main__":
    main()
