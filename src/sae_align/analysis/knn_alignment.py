from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


def cosine_knn_indices(x: np.ndarray, k: int = 10, batch_size: int = 512) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("Expected [n_samples, n_features] embeddings.")
    n = x.shape[0]
    if n < 2:
        raise ValueError("kNN requires at least two samples.")
    kk = min(int(k), n - 1)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    xn = x / np.maximum(norm, 1e-8)
    out = np.zeros((n, kk), dtype=np.int64)
    for start in range(0, n, int(batch_size)):
        end = min(n, start + int(batch_size))
        sim = xn[start:end] @ xn.T
        rows = np.arange(start, end)
        sim[np.arange(end - start), rows] = -np.inf
        part = np.argpartition(-sim, kth=kk - 1, axis=1)[:, :kk]
        part_scores = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        out[start:end] = np.take_along_axis(part, order, axis=1)
    return out


def neighbor_overlap(a: np.ndarray, b: np.ndarray, k: int | None = None, query_mask: np.ndarray | None = None) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    kk = min(a.shape[1], b.shape[1]) if k is None else int(k)
    rows = np.arange(a.shape[0]) if query_mask is None else np.nonzero(query_mask)[0]
    if rows.size == 0 or kk == 0:
        return float("nan")
    scores = []
    for i in rows:
        scores.append(len(set(a[i, :kk].tolist()) & set(b[i, :kk].tolist())) / float(kk))
    return float(np.mean(scores))


def label_purity(neighbors: np.ndarray, labels: np.ndarray, query_mask: np.ndarray | None = None) -> float:
    labels = np.asarray(labels)
    rows = np.arange(neighbors.shape[0]) if query_mask is None else np.nonzero(query_mask)[0]
    if rows.size == 0:
        return float("nan")
    scores = []
    for i in rows:
        scores.append(float(np.mean(labels[neighbors[i]] == labels[i])))
    return float(np.mean(scores))


def continuous_neighbor_mae(neighbors: np.ndarray, values: np.ndarray, query_mask: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=np.float32)
    rows = np.arange(neighbors.shape[0]) if query_mask is None else np.nonzero(query_mask)[0]
    if rows.size == 0:
        return float("nan")
    scores = []
    for i in rows:
        scores.append(float(np.mean(np.abs(values[neighbors[i]] - values[i]))))
    return float(np.mean(scores))


def pairwise_overlap_matrix(neighbor_sets: Mapping[str, np.ndarray], k: int | None = None) -> tuple[list[str], np.ndarray]:
    channels = list(neighbor_sets.keys())
    mat = np.zeros((len(channels), len(channels)), dtype=np.float32)
    for i, a in enumerate(channels):
        for j, b in enumerate(channels):
            mat[i, j] = neighbor_overlap(neighbor_sets[a], neighbor_sets[b], k=k)
    return channels, mat


def stratified_knn_summary(
    neighbor_sets: Mapping[str, np.ndarray],
    *,
    action_type: np.ndarray,
    effect_bin: np.ndarray,
    world_delta: np.ndarray,
    strata: Mapping[str, np.ndarray],
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for channel, neighbors in neighbor_sets.items():
        for stratum, mask in strata.items():
            rows.append(
                {
                    "channel": channel,
                    "stratum": stratum,
                    "n_queries": int(np.asarray(mask, dtype=bool).sum()),
                    "action_type_purity": label_purity(neighbors, action_type, query_mask=mask),
                    "effect_bin_purity": label_purity(neighbors, effect_bin, query_mask=mask),
                    "world_delta_neighbor_mae": continuous_neighbor_mae(neighbors, world_delta, query_mask=mask),
                }
            )
    return rows


def pairwise_stratified_overlap_rows(
    neighbor_sets: Mapping[str, np.ndarray],
    pair_strata: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    *,
    k: int | None = None,
    control: str = "observed",
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    channels = list(neighbor_sets.keys())
    n = next(iter(neighbor_sets.values())).shape[0] if neighbor_sets else 0
    kk = min(next(iter(neighbor_sets.values())).shape[1], int(k)) if neighbor_sets and k is not None else (
        next(iter(neighbor_sets.values())).shape[1] if neighbor_sets else 0
    )
    chance = float(kk / max(1, n - 1)) if n > 1 else float("nan")
    for i, a in enumerate(channels):
        for b in channels[i + 1 :]:
            strata = pair_strata[(a, b)]
            for stratum, mask in strata.items():
                mask = np.asarray(mask, dtype=bool)
                overlap = neighbor_overlap(neighbor_sets[a], neighbor_sets[b], k=k, query_mask=mask)
                rows.append(
                    {
                        "control": control,
                        "channel_a": a,
                        "channel_b": b,
                        "stratum": stratum,
                        "n_queries": int(mask.sum()),
                        "overlap": overlap,
                        "random_expected_overlap": chance,
                        "chance_adjusted_overlap": float(overlap - chance) if np.isfinite(overlap) else float("nan"),
                    }
                )
    return rows
