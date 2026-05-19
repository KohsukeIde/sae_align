#!/usr/bin/env python
"""Stage C0.5 oracle-positive detector smoke.

This script tests whether a lightweight classifier can detect useful weighting
signals at all. It is not a PSP/Dreamer baseline and not a final Stage C method.
The first gate is oracle-positive: oracle event/change weights should beat
uniform before observability weights are interpreted.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from train_stagec0_prediction import (
    STAGEB_REFERENCE_SIGNAL,
    binary_oracle_weights,
    build_features,
    detectability_matrix,
    effective_sample_size,
    event_targets,
    load_bundle,
    mix_observability_ranks,
    normalize_weights,
    percentile_rank,
    percentile_rank_columns,
    reconstruction_targets,
    split_states,
    validate_bundle,
)


DEFAULT_METHODS = (
    "uniform",
    "change_mask",
    "observability",
    "shuffled_observability",
    "oracle_event_class_weight",
    "oracle_event_sample_weight",
    "oracle_changed_class_weight",
    "oracle_changed_sample_weight",
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
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    lr: float,
    epochs: int,
    l2: float,
) -> Tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = np.maximum(weights, 1e-8)
    weights = weights / (np.mean(weights) + 1e-12)
    w = np.zeros(X.shape[1], dtype=np.float64)
    b = float(np.log((np.mean(y) + 1e-4) / (1.0 - np.mean(y) + 1e-4)))
    denom = float(np.sum(weights)) + 1e-12
    for _ in range(int(epochs)):
        p = sigmoid(X @ w + b)
        err = (p - y) * weights
        grad_w = (X.T @ err) / denom + float(l2) * w
        grad_b = float(np.sum(err) / denom)
        w -= float(lr) * grad_w
        b -= float(lr) * grad_b
    return w.astype(np.float32), float(b)


def predict_logistic(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(np.asarray(X, dtype=np.float32) @ w.astype(np.float32) + float(b)).astype(np.float32)


def average_ranks_ascending(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sorted_x = x[order]
    start = 0
    while start < x.size:
        end = start + 1
        while end < x.size and sorted_x[end] == sorted_x[start]:
            end += 1
        avg = 0.5 * float(start + end - 1) + 1.0
        ranks[order[start:end]] = avg
        start = end
    return ranks


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(bool).reshape(-1)
    s = np.asarray(y_score, dtype=np.float64).reshape(-1)
    n_pos = int(np.sum(y))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks_ascending(s)
    rank_sum_pos = float(np.sum(ranks[y]))
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(bool).reshape(-1)
    s = np.asarray(y_score, dtype=np.float64).reshape(-1)
    n_pos = int(np.sum(y))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted).astype(np.float64)
    rank = np.arange(1, y.size + 1, dtype=np.float64)
    precision = tp / rank
    return float(np.sum(precision[y_sorted]) / n_pos)


def binary_metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y = np.asarray(y_true).astype(bool).reshape(-1)
    pred = np.asarray(y_score).reshape(-1) >= float(threshold)
    tp = float(np.sum(y & pred))
    tn = float(np.sum(~y & ~pred))
    fp = float(np.sum(~y & pred))
    fn = float(np.sum(y & ~pred))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    balanced = 0.5 * (recall + specificity)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "balanced_accuracy": float(balanced),
    }


def choose_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    s = np.asarray(y_score, dtype=np.float64).reshape(-1)
    thresholds = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 99)))
    if thresholds.size == 0:
        return 0.5
    best_t = float(thresholds[0])
    best_f1 = -1.0
    for t in thresholds:
        f1 = binary_metrics_at_threshold(y_true, s, float(t))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def classifier_metrics(
    y_val: np.ndarray,
    score_val: np.ndarray,
    y_test: np.ndarray,
    score_test: np.ndarray,
    *,
    prefix: str,
) -> Dict[str, float]:
    threshold = choose_threshold_by_f1(y_val, score_val)
    at = binary_metrics_at_threshold(y_test, score_test, threshold)
    return {
        f"{prefix}_threshold": float(threshold),
        f"{prefix}_f1": at["f1"],
        f"{prefix}_precision": at["precision"],
        f"{prefix}_recall": at["recall"],
        f"{prefix}_balanced_accuracy": at["balanced_accuracy"],
        f"{prefix}_auprc": auprc(y_test, score_test),
        f"{prefix}_auroc": auroc(y_test, score_test),
    }


def method_weights(
    method: str,
    *,
    observability_score: np.ndarray,
    changed_ratio: np.ndarray,
    event_present: np.ndarray,
    changed_any: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if method == "uniform":
        return np.ones(event_present.size, dtype=np.float32)
    if method == "change_mask":
        return normalize_weights(changed_ratio, alpha)
    if method == "observability":
        return normalize_weights(observability_score, alpha)
    if method == "shuffled_observability":
        return normalize_weights(rng.permutation(observability_score), alpha)
    if method in {"oracle_event_class_weight", "oracle_event_sample_weight"}:
        return binary_oracle_weights(event_present, alpha)
    if method in {"oracle_changed_class_weight", "oracle_changed_sample_weight"}:
        return binary_oracle_weights(changed_any, alpha)
    raise ValueError(f"Unknown method {method!r}")


def summarize(rows: List[Mapping[str, object]]) -> List[Dict[str, object]]:
    keys = ["method", "alpha"]
    metrics = [
        "event_f1",
        "event_auprc",
        "event_auroc",
        "event_balanced_accuracy",
        "event_ood_f1",
        "event_ood_auprc",
        "event_ood_auroc",
        "event_ood_balanced_accuracy",
    ]
    groups = sorted({(str(r["method"]), float(r["alpha"])) for r in rows})
    uniform_by_alpha: Dict[float, Mapping[str, object]] = {}
    for alpha in sorted({float(r["alpha"]) for r in rows}):
        subset = [r for r in rows if str(r["method"]) == "uniform" and float(r["alpha"]) == alpha]
        if subset:
            uniform_by_alpha[alpha] = {
                m: float(np.nanmean([float(r[m]) for r in subset])) for m in metrics
            }
    out: List[Dict[str, object]] = []
    for method, alpha in groups:
        subset = [r for r in rows if str(r["method"]) == method and float(r["alpha"]) == alpha]
        row: Dict[str, object] = {"method": method, "alpha": alpha, "n_runs": len(subset)}
        for metric in metrics:
            vals = np.asarray([float(r[metric]) for r in subset], dtype=np.float64)
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_std"] = float(np.nanstd(vals))
            if alpha in uniform_by_alpha:
                delta = float(np.nanmean(vals) - float(uniform_by_alpha[alpha][metric]))
                row[f"{metric}_delta_vs_uniform"] = delta
                row[f"{metric}_delta_minus_stageb_ref"] = delta - STAGEB_REFERENCE_SIGNAL
        for key in [
            "weight_ess_train",
            "top_decile_weight_mass_train",
            "positive_event_weight_mass_train",
            "negative_event_weight_mass_train",
        ]:
            vals = np.asarray([float(r[key]) for r in subset], dtype=np.float64)
            row[f"{key}_mean"] = float(np.nanmean(vals))
        out.append(row)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage C0.5 oracle-positive classifier detector.")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--input-channels", nargs="*", default=["rgb"])
    p.add_argument("--target-channel", default="rgb")
    p.add_argument("--target-kind", choices=["event_present", "changed_any"], default="event_present")
    p.add_argument("--event-channel", default="event_response")
    p.add_argument("--observability-channels", nargs="*", default=["rgb", "range"])
    p.add_argument("--observability-mix", choices=["geom", "mean", "product", "min"], default="geom")
    p.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    p.add_argument("--alphas", nargs="*", type=float, default=[0.0, 1.0, 4.0, 16.0, 64.0])
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--channel-dim", type=int, default=128)
    p.add_argument("--changed-threshold", type=float, default=1e-4)
    p.add_argument("--event-threshold", type=float, default=1e-4)
    p.add_argument("--ood-noise-std", type=float, default=0.35)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--max-samples", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    reports = out / "reports"
    ensure_dir(reports)

    bundle = load_bundle(Path(args.data))
    validate_bundle(bundle, input_channels=args.input_channels, observability_channels=args.observability_channels)

    n_dense = bundle.dense_indices.size
    if int(args.max_samples) > 0 and int(args.max_samples) < n_dense:
        rng = np.random.default_rng(123)
        selection = np.sort(rng.choice(np.arange(n_dense), size=int(args.max_samples), replace=False))
        arrays = dict(bundle.arrays)
        for key in list(arrays.keys()):
            if (key.startswith("delta_") or key.startswith("obs0_")) and arrays[key].shape[0] == n_dense:
                arrays[key] = arrays[key][selection]
        for key in ("static_obs_sample_indices", "delta_sample_indices"):
            if key in arrays and arrays[key].shape[0] == n_dense:
                arrays[key] = arrays[key][selection]
        from train_stagec0_prediction import DatasetBundle

        bundle = DatasetBundle(
            arrays=arrays,
            dense_indices=bundle.dense_indices[selection],
            full_indices=bundle.full_indices[selection],
            state_ids=bundle.state_ids[selection],
            action_array=bundle.action_array[selection],
            action_type=bundle.action_type[selection],
        )

    y_event_multi = event_targets(
        bundle,
        args.event_channel,
        args.event_threshold,
        allow_world_delta_fallback=False,
    )
    y_event = np.any(y_event_multi > 0.5, axis=1).astype(np.float32)
    y_rec = reconstruction_targets(bundle, args.target_channel)
    y_changed = (np.abs(y_rec) > float(args.changed_threshold)).astype(np.float32)
    changed_any = np.any(y_changed > 0.5, axis=1).astype(np.float32)
    changed_ratio = np.mean(y_changed, axis=1).astype(np.float32)
    detect_matrix = detectability_matrix(bundle, args.observability_channels)

    rows: List[Dict[str, object]] = []
    for seed in args.seeds:
        train_mask, val_mask, test_mask = split_states(bundle.state_ids, seed, 0.60, 0.20)
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
        obs_train_ranks = percentile_rank_columns(detect_matrix[train_mask])
        obs_train = mix_observability_ranks(obs_train_ranks, mix=args.observability_mix)
        rng = np.random.default_rng(seed + 5003)
        for alpha in args.alphas:
            for method in args.methods:
                weights = method_weights(
                    method,
                    observability_score=obs_train,
                    changed_ratio=changed_ratio[train_mask],
                    event_present=y_event[train_mask],
                    changed_any=changed_any[train_mask],
                    alpha=float(alpha),
                    rng=rng,
                )
                y_target = y_event if args.target_kind == "event_present" else changed_any
                w, b = fit_logistic(
                    X_train,
                    y_target[train_mask],
                    weights,
                    lr=args.lr,
                    epochs=args.epochs,
                    l2=args.l2,
                )
                score_val = predict_logistic(X_val, w, b)
                score_test = predict_logistic(X_test, w, b)
                score_ood = predict_logistic(X_test_ood, w, b)
                row: Dict[str, object] = {
                    "seed": seed,
                    "method": method,
                    "alpha": float(alpha),
                    "target_kind": args.target_kind,
                    "input_channels": " ".join(args.input_channels),
                    "observability_mix": args.observability_mix,
                    "event_positive_rate_train": float(np.mean(y_event[train_mask])),
                    "event_positive_rate_val": float(np.mean(y_event[val_mask])),
                    "event_positive_rate_test": float(np.mean(y_event[test_mask])),
                    "changed_positive_rate_train": float(np.mean(changed_any[train_mask])),
                    "target_positive_rate_train": float(np.mean(y_target[train_mask])),
                    "target_positive_rate_val": float(np.mean(y_target[val_mask])),
                    "target_positive_rate_test": float(np.mean(y_target[test_mask])),
                    "weight_mean_train": float(np.mean(weights)),
                    "weight_std_train": float(np.std(weights)),
                    "weight_ess_train": float(effective_sample_size(weights)),
                    "top_decile_weight_mass_train": float(
                        np.sum(np.sort(weights)[-max(1, weights.size // 10) :])
                        / (np.sum(weights) + 1e-12)
                    ),
                    "positive_event_weight_mass_train": float(
                        np.sum(weights[y_event[train_mask] > 0.5]) / (np.sum(weights) + 1e-12)
                    ),
                    "negative_event_weight_mass_train": float(
                        np.sum(weights[y_event[train_mask] <= 0.5]) / (np.sum(weights) + 1e-12)
                    ),
                }
                row.update(classifier_metrics(y_target[val_mask], score_val, y_target[test_mask], score_test, prefix="event"))
                row.update(
                    classifier_metrics(
                        y_target[val_mask],
                        score_val,
                        y_target[test_mask],
                        score_ood,
                        prefix="event_ood",
                    )
                )
                rows.append(row)

    summary = summarize(rows)
    write_csv(reports / "stagec05_results.csv", rows)
    write_csv(reports / "stagec05_summary.csv", summary)

    best = sorted(
        [
            r
            for r in summary
            if str(r["method"]) != "uniform" and np.isfinite(float(r.get("event_auprc_delta_vs_uniform", float("nan"))))
        ],
        key=lambda r: max(
            float(r.get("event_auprc_delta_vs_uniform", -999.0)),
            float(r.get("event_f1_delta_vs_uniform", -999.0)),
            float(r.get("event_ood_auprc_delta_vs_uniform", -999.0)),
        ),
        reverse=True,
    )
    def summary_row(method: str, alpha: float | None = None) -> Mapping[str, object]:
        for row in summary:
            if str(row["method"]) != method:
                continue
            if alpha is None or abs(float(row["alpha"]) - float(alpha)) < 1e-12:
                return row
        return {}

    best_oracle = [r for r in best if str(r["method"]).startswith("oracle_event")]
    best_oracle_changed = [r for r in best if str(r["method"]).startswith("oracle_changed")]
    target_oracle_prefix = "oracle_event" if args.target_kind == "event_present" else "oracle_changed"
    best_target_oracle = [r for r in best if str(r["method"]).startswith(target_oracle_prefix)]
    best_obs = [r for r in best if str(r["method"]) == "observability"]
    decision = {
        "stage": "C0.5",
        "data": str(args.data),
        "input_channels": list(args.input_channels),
        "target_kind": args.target_kind,
        "n_result_rows": len(rows),
        "best_nonuniform": best[0] if best else {},
        "best_oracle_event": best_oracle[0] if best_oracle else {},
        "best_oracle_changed": best_oracle_changed[0] if best_oracle_changed else {},
        "target_oracle_prefix": target_oracle_prefix,
        "best_target_oracle": best_target_oracle[0] if best_target_oracle else {},
        "best_observability": best_obs[0] if best_obs else {},
        "target_oracle_best_delta_ge_0p10": bool(
            best_target_oracle
            and max(
                float(best_target_oracle[0].get("event_auprc_delta_vs_uniform", -999.0)),
                float(best_target_oracle[0].get("event_f1_delta_vs_uniform", -999.0)),
                float(best_target_oracle[0].get("event_ood_auprc_delta_vs_uniform", -999.0)),
            )
            >= 0.10
        ),
        "oracle_event_best_delta_ge_0p10": bool(
            best_oracle
            and max(
                float(best_oracle[0].get("event_auprc_delta_vs_uniform", -999.0)),
                float(best_oracle[0].get("event_f1_delta_vs_uniform", -999.0)),
                float(best_oracle[0].get("event_ood_auprc_delta_vs_uniform", -999.0)),
            )
            >= 0.10
        ),
        "observability_best_delta_ge_0p10": bool(
            best_obs
            and max(
                float(best_obs[0].get("event_auprc_delta_vs_uniform", -999.0)),
                float(best_obs[0].get("event_f1_delta_vs_uniform", -999.0)),
                float(best_obs[0].get("event_ood_auprc_delta_vs_uniform", -999.0)),
            )
            >= 0.10
        ),
        "uniform_event_auprc_mean_alpha0": summary_row("uniform", 0.0).get("event_auprc_mean"),
    }
    write_json(reports / "stagec05_decision_summary.json", decision)
    print(f"Wrote Stage C0.5 reports to {reports}")


if __name__ == "__main__":
    main()
