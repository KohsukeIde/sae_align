"""Small NumPy helpers for Action-IV experiments.

These utilities intentionally avoid sklearn/scipy so the repo stays lightweight.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))).astype(np.float32)


def average_rank(values: np.ndarray) -> np.ndarray:
    """Tie-aware [0, 1] average rank."""
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
    train = np.asarray(train_values, dtype=np.float64).reshape(-1)
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if train.size <= 1:
        return np.zeros_like(vals, dtype=np.float32)
    train_sorted = np.sort(train, kind="mergesort")
    left = np.searchsorted(train_sorted, vals, side="left")
    right = np.searchsorted(train_sorted, vals, side="right")
    mid = 0.5 * (left + right - 1)
    return np.clip(mid / float(train.size - 1), 0.0, 1.0).astype(np.float32)


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
            ap += (tp_seen / max(total_seen, 1)) * group_pos / pos
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
    qs = np.linspace(0.01, 0.99, 99)
    thresholds = np.unique(np.quantile(scores, qs))
    best_t = float(thresholds[0]) if thresholds.size else 0.5
    best_f1 = -1.0
    for t in thresholds:
        _, _, f1 = precision_recall_f1(y_true, scores, float(t))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def metric_dict(prefix: str, y: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    precision, recall, f1 = precision_recall_f1(y, score, threshold)
    y_bool = np.asarray(y).astype(bool).reshape(-1)
    pred = np.asarray(score).reshape(-1) >= float(threshold)
    tp = float(np.sum(y_bool & pred))
    tn = float(np.sum(~y_bool & ~pred))
    fp = float(np.sum(~y_bool & pred))
    fn = float(np.sum(y_bool & ~pred))
    tpr = tp / (tp + fn + 1e-12)
    tnr = tn / (tn + fp + 1e-12)
    return {
        f"{prefix}_auprc": average_precision(y, score),
        f"{prefix}_auroc": auroc_score(y, score),
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_balanced_acc": float(0.5 * (tpr + tnr)),
        f"{prefix}_positive_rate": float(np.mean(y_bool)),
        f"{prefix}_threshold": float(threshold),
    }


def train_val_test_by_state(state_ids: np.ndarray, seed: int, train_frac: float = 0.65, val_frac: float = 0.15):
    rng = np.random.default_rng(int(seed))
    unique = np.unique(state_ids.astype(np.int64))
    rng.shuffle(unique)
    n = unique.size
    n_train = max(1, int(round(train_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_set = set(unique[:n_train].tolist())
    val_set = set(unique[n_train:n_train + n_val].tolist())
    train = np.array([int(s) in train_set for s in state_ids], dtype=bool)
    val = np.array([int(s) in val_set for s in state_ids], dtype=bool)
    test = ~(train | val)
    return train, val, test


def train_val_test_by_state_with_labels(
    state_ids: np.ndarray,
    y: np.ndarray,
    seed: int,
    train_frac: float = 0.65,
    val_frac: float = 0.15,
    min_pos: int = 1,
    min_neg: int = 1,
    max_tries: int = 128,
):
    """State-group split with a simple positive/negative sanity retry.

    The split unit remains state_id; labels are used only to avoid degenerate
    train/validation/test partitions for sparse binary tasks.
    """
    labels = np.asarray(y).astype(bool).reshape(-1)
    for offset in range(int(max_tries)):
        train, val, test = train_val_test_by_state(state_ids, int(seed) + offset, train_frac, val_frac)
        ok = True
        for mask in (train, val, test):
            pos = int(np.sum(labels[mask]))
            neg = int(np.sum(~labels[mask]))
            if pos < int(min_pos) or neg < int(min_neg):
                ok = False
                break
        if ok:
            return train, val, test
    return train_val_test_by_state(state_ids, seed, train_frac, val_frac)


def standardize_train_apply(x_train: np.ndarray, x_all: np.ndarray):
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x_train - mean) / std).astype(np.float32), ((x_all - mean) / std).astype(np.float32), mean, std


def random_project(x: np.ndarray, out_dim: int, seed: int):
    x = np.asarray(x, dtype=np.float32).reshape(x.shape[0], -1)
    if out_dim <= 0 or out_dim >= x.shape[1]:
        return x.astype(np.float32), None
    rng = np.random.default_rng(int(seed))
    proj = rng.normal(0.0, 1.0 / math.sqrt(float(out_dim)), size=(x.shape[1], out_dim)).astype(np.float32)
    return (x @ proj).astype(np.float32), proj


def fit_logistic(X: np.ndarray, y: np.ndarray, weights: np.ndarray, epochs: int = 120, lr: float = 0.2, l2: float = 1e-4):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    w = np.maximum(w, 1e-6)
    w = w / (np.mean(w) + 1e-12)
    theta = np.zeros(X.shape[1], dtype=np.float64)
    bias = 0.0
    denom = float(w.sum()) + 1e-12
    for _ in range(int(epochs)):
        p = sigmoid(X @ theta + bias).astype(np.float64)
        err = (p - y) * w
        grad = (X.T @ err) / denom + float(l2) * theta
        grad_b = float(np.sum(err) / denom)
        theta -= float(lr) * grad
        bias -= float(lr) * grad_b
    return theta.astype(np.float32), float(bias)


def predict_logistic(X: np.ndarray, theta: np.ndarray, bias: float) -> np.ndarray:
    return sigmoid(np.asarray(X, dtype=np.float32) @ theta.astype(np.float32) + float(bias))


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def recall_at_k(query: np.ndarray, target: np.ndarray, k: int) -> float:
    q = l2_normalize(query)
    t = l2_normalize(target)
    sim = q @ t.T
    # positive for row i is column i
    top = np.argpartition(-sim, kth=min(k, sim.shape[1]-1), axis=1)[:, :k]
    hits = np.array([i in top[i] for i in range(sim.shape[0])], dtype=np.float32)
    return float(np.mean(hits))
