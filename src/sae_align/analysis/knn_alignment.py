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


def restricted_cosine_knn_indices(
    x: np.ndarray,
    restrict_labels: np.ndarray,
    k: int = 10,
    batch_size: int = 512,
) -> np.ndarray:
    """Cosine kNN with candidates restricted to matching labels.

    Returned neighbor IDs are always in the original row coordinates of ``x``.
    Rows with fewer than ``k`` eligible candidates are padded with -1.
    """
    x = np.asarray(x, dtype=np.float32)
    labels = np.asarray(restrict_labels)
    if x.ndim != 2:
        raise ValueError("Expected [n_samples, n_features] embeddings.")
    if labels.shape[:1] != x.shape[:1]:
        raise ValueError("restrict_labels must have one label per embedding row.")
    n = x.shape[0]
    if n < 2:
        raise ValueError("kNN requires at least two samples.")
    kk = min(int(k), n - 1)
    out = np.full((n, kk), -1, dtype=np.int64)
    for label in np.unique(labels):
        group = np.nonzero(labels == label)[0]
        if group.size < 2:
            continue
        local_k = min(kk, int(group.size) - 1)
        local_neighbors = cosine_knn_indices(x[group], k=local_k, batch_size=batch_size)
        out[group, :local_k] = group[local_neighbors]
    return out


def restricted_candidate_counts(restrict_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(restrict_labels)
    counts = {label: int(np.sum(labels == label)) for label in np.unique(labels)}
    return np.asarray([max(0, counts[label] - 1) for label in labels], dtype=np.int64)


def excluded_candidate_counts(exclude_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(exclude_labels)
    n = int(labels.shape[0])
    counts = {label: int(np.sum(labels == label)) for label in np.unique(labels)}
    return np.asarray([max(0, n - counts[label]) for label in labels], dtype=np.int64)


def excluded_cosine_knn_indices(
    x: np.ndarray,
    exclude_labels: np.ndarray,
    k: int = 10,
    batch_size: int = 512,
) -> np.ndarray:
    """Cosine kNN while excluding candidates with the same label as the query."""
    x = np.asarray(x, dtype=np.float32)
    labels = np.asarray(exclude_labels)
    if x.ndim != 2:
        raise ValueError("Expected [n_samples, n_features] embeddings.")
    if labels.shape[:1] != x.shape[:1]:
        raise ValueError("exclude_labels must have one label per embedding row.")
    n = x.shape[0]
    if n < 2:
        raise ValueError("kNN requires at least two samples.")
    kk = min(int(k), n - 1)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    xn = x / np.maximum(norm, 1e-8)
    out = np.full((n, kk), -1, dtype=np.int64)
    for start in range(0, n, int(batch_size)):
        end = min(n, start + int(batch_size))
        sim = xn[start:end] @ xn.T
        same = labels[None, :] == labels[start:end, None]
        sim[same] = -np.inf
        for local_row in range(end - start):
            finite = np.isfinite(sim[local_row])
            local_k = min(kk, int(finite.sum()))
            if local_k == 0:
                continue
            part = np.argpartition(-sim[local_row], kth=local_k - 1)[:local_k]
            order = np.argsort(-sim[local_row, part])
            out[start + local_row, :local_k] = part[order]
    return out


def shuffled_assignment_knn_indices(
    x: np.ndarray,
    permutation: np.ndarray,
    k: int = 10,
    batch_size: int = 512,
) -> np.ndarray:
    """kNN after randomly assigning embeddings to original sample rows.

    This is a destructive shuffled-embedding control: row ``i`` keeps original
    sample ID ``i`` for masks/strata, but receives embedding ``x[permutation[i]]``.
    The returned neighbor IDs are therefore already in original row coordinates.
    """
    x = np.asarray(x, dtype=np.float32)
    permutation = np.asarray(permutation, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError("Expected [n_samples, n_features] embeddings.")
    if permutation.shape != (x.shape[0],):
        raise ValueError("permutation must have one entry per embedding row.")
    if set(permutation.tolist()) != set(range(x.shape[0])):
        raise ValueError("permutation must contain each row index exactly once.")
    return cosine_knn_indices(x[permutation], k=k, batch_size=batch_size)


def action_confound_features(action_array: np.ndarray, action_type: np.ndarray) -> np.ndarray:
    action_array = np.asarray(action_array, dtype=np.float32)
    if action_array.ndim == 1:
        action_array = action_array[:, None]
    else:
        action_array = action_array.reshape(action_array.shape[0], -1)
    action_type = np.asarray(action_type).astype(str)
    if action_type.shape[:1] != action_array.shape[:1]:
        raise ValueError("action_type must have one label per action row.")
    names = sorted(set(action_type.tolist()))
    one_hot = np.zeros((action_type.shape[0], len(names)), dtype=np.float32)
    index = {name: i for i, name in enumerate(names)}
    for i, label in enumerate(action_type):
        one_hot[i, index[label]] = 1.0
    x = np.concatenate([action_array, one_hot], axis=1)
    return ((x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)).astype(np.float32)


def residualize_embeddings(embeddings: np.ndarray, confounds: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    confounds = np.asarray(confounds, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("Expected [n_samples, n_features] embeddings.")
    if confounds.ndim != 2:
        raise ValueError("Expected [n_samples, n_confounds] confounds.")
    if embeddings.shape[0] != confounds.shape[0]:
        raise ValueError("Embeddings and confounds must have the same row count.")
    design = np.concatenate(
        [confounds, np.ones((confounds.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    coef, *_ = np.linalg.lstsq(design.astype(np.float64), embeddings.astype(np.float64), rcond=None)
    resid = embeddings.astype(np.float64) - design.astype(np.float64) @ coef
    return resid.astype(np.float32)


def neighbor_overlap_stats(
    a: np.ndarray,
    b: np.ndarray,
    k: int | None = None,
    query_mask: np.ndarray | None = None,
) -> Dict[str, float | int]:
    a = np.asarray(a)
    b = np.asarray(b)
    kk = min(a.shape[1], b.shape[1]) if k is None else int(k)
    rows = np.arange(a.shape[0]) if query_mask is None else np.nonzero(query_mask)[0]
    if rows.size == 0 or kk == 0:
        return {"overlap": float("nan"), "n_valid_queries": 0, "mean_effective_k": float("nan")}
    scores = []
    effective_ks = []
    for i in rows:
        a_row = [int(v) for v in a[i, :kk].tolist() if int(v) >= 0]
        b_row = [int(v) for v in b[i, :kk].tolist() if int(v) >= 0]
        denom = min(kk, len(a_row), len(b_row))
        if denom == 0:
            continue
        scores.append(len(set(a_row) & set(b_row)) / float(denom))
        effective_ks.append(denom)
    if not scores:
        return {"overlap": float("nan"), "n_valid_queries": 0, "mean_effective_k": float("nan")}
    return {
        "overlap": float(np.mean(scores)),
        "n_valid_queries": int(len(scores)),
        "mean_effective_k": float(np.mean(effective_ks)),
    }


def neighbor_overlap(a: np.ndarray, b: np.ndarray, k: int | None = None, query_mask: np.ndarray | None = None) -> float:
    return float(neighbor_overlap_stats(a, b, k=k, query_mask=query_mask)["overlap"])


def label_purity(neighbors: np.ndarray, labels: np.ndarray, query_mask: np.ndarray | None = None) -> float:
    labels = np.asarray(labels)
    rows = np.arange(neighbors.shape[0]) if query_mask is None else np.nonzero(query_mask)[0]
    if rows.size == 0:
        return float("nan")
    scores = []
    for i in rows:
        row = neighbors[i]
        row = row[row >= 0]
        if row.size == 0:
            continue
        scores.append(float(np.mean(labels[row] == labels[i])))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def continuous_neighbor_mae(neighbors: np.ndarray, values: np.ndarray, query_mask: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=np.float32)
    rows = np.arange(neighbors.shape[0]) if query_mask is None else np.nonzero(query_mask)[0]
    if rows.size == 0:
        return float("nan")
    scores = []
    for i in rows:
        row = neighbors[i]
        row = row[row >= 0]
        if row.size == 0:
            continue
        scores.append(float(np.mean(np.abs(values[row] - values[i]))))
    if not scores:
        return float("nan")
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
    candidate_counts: np.ndarray | None = None,
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    channels = list(neighbor_sets.keys())
    n = next(iter(neighbor_sets.values())).shape[0] if neighbor_sets else 0
    kk = min(next(iter(neighbor_sets.values())).shape[1], int(k)) if neighbor_sets and k is not None else (
        next(iter(neighbor_sets.values())).shape[1] if neighbor_sets else 0
    )
    candidate_counts_arr = None if candidate_counts is None else np.asarray(candidate_counts, dtype=np.int64)
    if candidate_counts_arr is not None and candidate_counts_arr.shape[:1] != (n,):
        raise ValueError("candidate_counts must have one value per query row.")
    chance = float(kk / max(1, n - 1)) if n > 1 and candidate_counts_arr is None else float("nan")
    for i, a in enumerate(channels):
        for b in channels[i + 1 :]:
            strata = pair_strata[(a, b)]
            for stratum, mask in strata.items():
                mask = np.asarray(mask, dtype=bool)
                stats = neighbor_overlap_stats(neighbor_sets[a], neighbor_sets[b], k=k, query_mask=mask)
                overlap = float(stats["overlap"])
                row_chance = chance
                if candidate_counts_arr is not None:
                    eligible = candidate_counts_arr[mask]
                    eligible = eligible[eligible > 0]
                    if eligible.size:
                        row_chance = float(np.mean(np.minimum(kk, eligible) / eligible))
                rows.append(
                    {
                        "control": control,
                        "channel_a": a,
                        "channel_b": b,
                        "stratum": stratum,
                        "n_queries": int(mask.sum()),
                        "n_valid_queries": int(stats["n_valid_queries"]),
                        "mean_effective_k": float(stats["mean_effective_k"]),
                        "overlap": overlap,
                        "random_expected_overlap": row_chance,
                        "chance_adjusted_overlap": float(overlap - row_chance)
                        if np.isfinite(overlap) and np.isfinite(row_chance)
                        else float("nan"),
                    }
                )
    return rows
