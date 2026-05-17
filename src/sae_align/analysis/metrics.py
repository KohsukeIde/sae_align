from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


def l1_detectability(delta: np.ndarray) -> np.ndarray:
    """Mean absolute effect magnitude per sample."""
    if delta.ndim == 1:
        return np.abs(delta)
    return np.mean(np.abs(delta.reshape(delta.shape[0], -1)), axis=1)


def percentile_threshold(values: np.ndarray, q: float) -> float:
    """Return percentile threshold.

    q is in [0, 1].  The threshold is computed on all values; callers may choose
    to pass non-null values if desired.
    """
    return float(np.quantile(values, q))


def binary_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(a, b).sum() / union)


def overlap_at_k(neigh_a: np.ndarray, neigh_b: np.ndarray, k: int) -> float:
    """Average kNN overlap.

    neigh_a/neigh_b: integer arrays [N, k].
    """
    scores = []
    for row_a, row_b in zip(neigh_a[:, :k], neigh_b[:, :k]):
        scores.append(len(set(row_a.tolist()).intersection(row_b.tolist())) / float(k))
    return float(np.mean(scores))


def knn_indices(x: np.ndarray, k: int = 10, max_points: int | None = None, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Brute-force kNN indices for small diagnostic runs.

    Returns sampled indices and neighbor indices in original coordinates.
    """
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    if max_points is not None and n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
    else:
        idx = np.arange(n)
    xs = x[idx].reshape(len(idx), -1).astype(np.float32)
    # Normalize to reduce scale artifacts.
    xs = xs - xs.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(xs, axis=1, keepdims=True) + 1e-8
    xs = xs / norms
    sim = xs @ xs.T
    np.fill_diagonal(sim, -np.inf)
    nn_local = np.argsort(-sim, axis=1)[:, :k]
    return idx, idx[nn_local]


def random_projection(x: np.ndarray, out_dim: int = 128, seed: int = 0) -> np.ndarray:
    x2 = x.reshape(x.shape[0], -1).astype(np.float32)
    # Standardize columns lightly.
    x2 = x2 - x2.mean(axis=0, keepdims=True)
    std = x2.std(axis=0, keepdims=True) + 1e-6
    x2 = x2 / std
    in_dim = x2.shape[1]
    if in_dim <= out_dim:
        return x2
    rng = np.random.default_rng(seed)
    proj = rng.normal(0.0, 1.0 / np.sqrt(out_dim), size=(in_dim, out_dim)).astype(np.float32)
    return x2 @ proj


def ridge_r2(x: np.ndarray, y: np.ndarray, alpha: float = 1.0, seed: int = 0, max_train: int = 2000) -> float:
    """Linear ridge R^2 using a train/test split.

    Inputs should already be moderate-dimensional; use random_projection first.
    """
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = rng.permutation(n)
    if n > max_train:
        idx = idx[:max_train]
        n = len(idx)
    split = max(2, int(0.8 * n))
    train, test = idx[:split], idx[split:]
    xtr, ytr = x[train], y[train]
    xte, yte = x[test], y[test]
    # Add bias.
    xtrb = np.concatenate([xtr, np.ones((xtr.shape[0], 1), dtype=xtr.dtype)], axis=1)
    xteb = np.concatenate([xte, np.ones((xte.shape[0], 1), dtype=xte.dtype)], axis=1)
    xtx = xtrb.T @ xtrb
    reg = alpha * np.eye(xtx.shape[0], dtype=np.float32)
    reg[-1, -1] = 0.0
    w = np.linalg.solve(xtx + reg, xtrb.T @ ytr)
    pred = xteb @ w
    ss_res = np.sum((yte - pred) ** 2)
    ss_tot = np.sum((yte - yte.mean(axis=0, keepdims=True)) ** 2) + 1e-8
    return float(1.0 - ss_res / ss_tot)


def auroc_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    """Small dependency-free AUROC."""
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U interpretation.
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)
    rank_pos = ranks[: len(pos)].sum()
    auc = (rank_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def best_f1(scores: np.ndarray, labels: np.ndarray, n_thresh: int = 101) -> Tuple[float, float]:
    labels = np.asarray(labels).astype(bool)
    qs = np.linspace(0.0, 1.0, n_thresh)
    thresholds = np.quantile(scores, qs)
    best = 0.0
    best_t = float(thresholds[0])
    for t in thresholds:
        pred = scores >= t
        tp = np.logical_and(pred, labels).sum()
        fp = np.logical_and(pred, ~labels).sum()
        fn = np.logical_and(~pred, labels).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best:
            best = float(f1)
            best_t = float(t)
    return best, best_t
