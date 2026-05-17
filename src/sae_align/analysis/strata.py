from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np


def diagnostic_only_channels(data: Mapping[str, np.ndarray]) -> set[str]:
    if "diagnostic_only_channels" not in data:
        return {"event", "event_response"}
    return {str(x) for x in np.asarray(data["diagnostic_only_channels"]).tolist()} | {"event", "event_response"}


def subset_stage0_vector(data: Mapping[str, np.ndarray], key: str, sample_indices: np.ndarray) -> np.ndarray:
    values = np.asarray(data[key])
    if values.shape[:1] == sample_indices.shape[:1]:
        return values
    return values[np.asarray(sample_indices, dtype=np.int64)]


def dense_delta_sample_indices(data: Mapping[str, np.ndarray], n_dense: int) -> np.ndarray:
    if "delta_sample_indices" in data:
        return np.asarray(data["delta_sample_indices"], dtype=np.int64)
    return np.arange(n_dense, dtype=np.int64)


def validate_dense_stage0_data(
    data: Mapping[str, np.ndarray],
    channels: Sequence[str],
    *,
    require_detect: bool = True,
) -> np.ndarray:
    if not channels:
        raise ValueError("No channels were selected.")
    first = f"delta_{channels[0]}"
    if first not in data:
        raise KeyError(f"Missing dense delta array {first}.")
    n_dense = int(np.asarray(data[first]).shape[0])
    sample_indices = dense_delta_sample_indices(data, n_dense)
    if sample_indices.ndim != 1:
        raise ValueError("delta_sample_indices must be one-dimensional.")
    if int(sample_indices.shape[0]) != n_dense:
        raise ValueError(f"delta_sample_indices has {sample_indices.shape[0]} rows but dense deltas have {n_dense}.")
    if n_dense == 0:
        raise ValueError("No dense delta samples are available.")
    n_full = int(np.asarray(data["world_delta"]).shape[0])
    if int(np.min(sample_indices)) < 0 or int(np.max(sample_indices)) >= n_full:
        raise ValueError("delta_sample_indices contains out-of-range sample IDs.")
    if len(np.unique(sample_indices)) != len(sample_indices):
        raise ValueError("delta_sample_indices contains duplicates.")
    for ch in channels:
        delta_key = f"delta_{ch}"
        if delta_key not in data:
            raise KeyError(f"Missing dense delta array {delta_key}.")
        if int(np.asarray(data[delta_key]).shape[0]) != n_dense:
            raise ValueError(f"{delta_key} row count does not match {first}.")
        if require_detect and f"detect_{ch}" not in data:
            raise KeyError(f"Missing detectability array detect_{ch}.")
    return sample_indices


def physical_nonnull_mask(world_delta: np.ndarray, q: float = 0.10) -> tuple[np.ndarray, float]:
    world_delta = np.asarray(world_delta, dtype=np.float32)
    if world_delta.size == 0:
        raise ValueError("world_delta is empty.")
    threshold = max(float(np.quantile(world_delta, float(q))), 1e-8)
    return world_delta >= threshold, threshold


def effect_bins(world_delta: np.ndarray, quantiles: Sequence[float] = (0.33, 0.66)) -> tuple[np.ndarray, np.ndarray]:
    world_delta = np.asarray(world_delta, dtype=np.float32)
    if world_delta.size == 0:
        raise ValueError("world_delta is empty.")
    cuts = np.quantile(world_delta, list(quantiles)).astype(np.float32)
    labels = np.searchsorted(cuts, world_delta, side="right").astype(np.int32)
    return labels, cuts


def named_strata(
    world_delta: np.ndarray,
    action_type: np.ndarray,
    *,
    physical_q: float = 0.10,
    effect_quantiles: Sequence[float] = (0.33, 0.66),
) -> Dict[str, np.ndarray]:
    world_delta = np.asarray(world_delta, dtype=np.float32)
    action_type = np.asarray(action_type).astype(str)
    physical, _ = physical_nonnull_mask(world_delta, q=physical_q)
    bins, _ = effect_bins(world_delta, quantiles=effect_quantiles)

    strata: Dict[str, np.ndarray] = {
        "all": np.ones(world_delta.shape[0], dtype=bool),
        "physical_nonnull": physical,
    }
    for b in sorted(set(bins.tolist())):
        strata[f"effect_bin_{int(b)}"] = bins == b
    for action in sorted(set(action_type.tolist())):
        strata[f"action_type_{action}"] = action_type == action
    return strata


def channel_blind_masks(
    data: Mapping[str, np.ndarray],
    channels: Sequence[str],
    sample_indices: np.ndarray,
    *,
    threshold_quantile: float = 0.10,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    """Compute dense-subset physical, blind, and regular masks per channel.

    Strata are defined from oracle world-state deltas and observation-level
    detectability only. Embeddings are intentionally not used.
    """
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    world_full = np.asarray(data["world_delta"], dtype=np.float32)
    eps_x = max(float(np.quantile(world_full, float(threshold_quantile))), 1e-8)
    physical_full = world_full >= eps_x
    physical = physical_full[sample_indices]
    blind: dict[str, np.ndarray] = {}
    regular: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {"world": eps_x}
    for ch in channels:
        det_full = np.asarray(data[f"detect_{ch}"], dtype=np.float32)
        base = det_full[physical_full]
        tau = max(float(np.quantile(base, float(threshold_quantile))) if base.size else 1e-8, 1e-8)
        det = det_full[sample_indices]
        blind[ch] = np.logical_and(physical, det <= tau)
        regular[ch] = np.logical_and(physical, det > tau)
        thresholds[ch] = tau
    return physical, blind, regular, thresholds


def pair_channel_strata(
    physical_nonnull: np.ndarray,
    blind: Mapping[str, np.ndarray],
    regular: Mapping[str, np.ndarray],
    a: str,
    b: str,
) -> dict[str, np.ndarray]:
    physical_nonnull = np.asarray(physical_nonnull, dtype=bool)
    physical_null = ~physical_nonnull
    blind_a = np.asarray(blind[a], dtype=bool)
    blind_b = np.asarray(blind[b], dtype=bool)
    regular_a = np.asarray(regular[a], dtype=bool)
    regular_b = np.asarray(regular[b], dtype=bool)
    return {
        "all": np.ones_like(physical_nonnull, dtype=bool),
        "physical_nonnull": physical_nonnull,
        "physical_null": physical_null,
        "regular_both": np.logical_and(regular_a, regular_b),
        "blind_a": blind_a,
        "blind_b": blind_b,
        "blind_either": np.logical_or(blind_a, blind_b),
        "blind_both": np.logical_and(blind_a, blind_b),
        "regular_a_blind_b": np.logical_and(regular_a, blind_b),
        "blind_a_regular_b": np.logical_and(blind_a, regular_b),
    }
