#!/usr/bin/env python
"""Stage B.3 framing-decision gates.

This script audits whether Stage B should stay with binary/continuous strata,
move toward action-coupling, or repair the metric before any framing decision.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_SCRIPT_DIR = _Path(__file__).resolve().parent
_REPO_SRC = _SCRIPT_DIR.parents[0] / "src"
for _p in (str(_SCRIPT_DIR), str(_REPO_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import csv
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from analyze_state_signature_knn import (
    bootstrap_cross_rows,
    bootstrap_rows,
    complete_state_action_matrix,
    cross_split_rows,
    dense_rows_for_sample_ids,
    load_npz,
    pairwise_stratified_overlap_rows,
    state_pair_strata,
    summarize_fraction_rows,
    validate_model_data_path,
    validate_model_fingerprint,
    validate_probe_trained_model,
)
from sae_align.analysis.knn_alignment import cosine_knn_indices
from sae_align.analysis.strata import diagnostic_only_channels, validate_dense_stage0_data, validate_dense_static_data
from sae_align.models import load_transition_encoders
from sae_align.utils.io import ensure_dir, save_json


NORMALIZATION_MODES = (
    "none",
    "probe_global_apply",
    "probe_action_type_apply",
    "split_global_diagnostic",
    "per_action_diagnostic",
    "per_action_type_diagnostic",
)
PRIMARY_NORMALIZATION_MODES = {"none", "probe_global_apply", "probe_action_type_apply"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--static-model", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--channels", nargs="*", default=None)
    p.add_argument("--probe-action-ids", nargs="*", type=int, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-states", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--detectability-quantile", type=float, default=0.10)
    p.add_argument("--regular-state-threshold", type=float, default=0.25)
    p.add_argument("--blind-state-threshold", type=float, default=0.25)
    p.add_argument("--physical-state-threshold", type=float, default=0.25)
    p.add_argument("--bootstrap-repeats", type=int, default=200)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--score-quantile-bins", type=int, default=5)
    p.add_argument("--normalization-modes", nargs="*", default=list(NORMALIZATION_MODES))
    p.add_argument("--allow-cross-data-model", action="store_true")
    p.add_argument("--allow-all-action-trained-model", action="store_true")
    return p.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(order.size, dtype=np.float32)
    if order.size <= 1:
        return np.zeros_like(x, dtype=np.float32)
    return ranks / float(order.size - 1)


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or float(np.std(x)) <= 1e-8 or float(np.std(y)) <= 1e-8:
        return float("nan"), float("nan"), int(x.size)
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rank01(x), rank01(y))[0, 1])
    return pearson, spearman, int(x.size)


def normalize_tensor(
    tensor: np.ndarray,
    *,
    mode: str,
    action_types: np.ndarray,
    reference: np.ndarray | None = None,
    reference_action_types: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(tensor, dtype=np.float32).copy()
    base_mode = mode.replace("_diagnostic", "")
    if base_mode == "probe_action_type_apply":
        base_mode = "per_action_type"
    if mode == "none":
        return x
    if base_mode == "per_action":
        return ((x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)).astype(np.float32)
    if base_mode == "split_global":
        return ((x - x.mean(axis=(0, 1), keepdims=True)) / (x.std(axis=(0, 1), keepdims=True) + 1e-6)).astype(np.float32)
    if base_mode == "probe_global_apply":
        ref = x if reference is None else np.asarray(reference, dtype=np.float32)
        return ((x - ref.mean(axis=(0, 1), keepdims=True)) / (ref.std(axis=(0, 1), keepdims=True) + 1e-6)).astype(np.float32)
    if base_mode == "per_action_type":
        out = x.copy()
        for label in np.unique(action_types.astype(str)):
            cols = action_types.astype(str) == str(label)
            if not np.any(cols):
                continue
            if reference is None:
                ref_block = out[:, cols, :]
            else:
                ref = np.asarray(reference, dtype=np.float32)
                ref_types = action_types if reference_action_types is None else reference_action_types
                ref_cols = ref_types.astype(str) == str(label)
                if not np.any(ref_cols):
                    continue
                ref_block = ref[:, ref_cols, :]
            out[:, cols, :] = (out[:, cols, :] - ref_block.mean(axis=(0, 1), keepdims=True)) / (
                ref_block.std(axis=(0, 1), keepdims=True) + 1e-6
            )
        return out.astype(np.float32)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def embedding_tensors(
    data: Mapping[str, np.ndarray],
    encoders: Mapping[str, object],
    channels: Sequence[str],
    *,
    prefix: str,
    dense_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    n_states, n_actions = dense_rows.shape
    flat = dense_rows.reshape(-1)
    out = {}
    for channel in channels:
        emb = encoders[channel].transform(data[f"{prefix}_{channel}"][flat])
        out[channel] = emb.reshape(n_states, n_actions, -1).astype(np.float32)
    return out


def features_from_tensors(
    tensors: Mapping[str, np.ndarray],
    channels: Sequence[str],
    positions: np.ndarray,
    *,
    mode: str,
    action_types: np.ndarray,
    reference_positions: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    out = {}
    for channel in channels:
        tensor = tensors[channel][:, positions, :]
        reference = None if reference_positions is None else tensors[channel][:, reference_positions, :]
        reference_types = None if reference_positions is None else action_types[reference_positions]
        normed = normalize_tensor(
            tensor,
            mode=mode,
            action_types=action_types[positions],
            reference=reference,
            reference_action_types=reference_types,
        )
        out[channel] = normed.reshape(normed.shape[0], -1).astype(np.float32)
    return out


def normalization_is_primary(mode: str) -> bool:
    return str(mode) in PRIMARY_NORMALIZATION_MODES


def normalization_is_transductive(mode: str) -> bool:
    return str(mode).endswith("_diagnostic")


def shuffled_features_from_tensors(
    tensors: Mapping[str, np.ndarray],
    channels: Sequence[str],
    positions: np.ndarray,
    *,
    mode: str,
    action_types: np.ndarray,
    reference_positions: np.ndarray | None = None,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for channel in channels:
        tensor = tensors[channel][:, positions, :].copy()
        for state_i in range(tensor.shape[0]):
            tensor[state_i] = tensor[state_i, rng.permutation(tensor.shape[1]), :]
        reference = None if reference_positions is None else tensors[channel][:, reference_positions, :]
        reference_types = None if reference_positions is None else action_types[reference_positions]
        normed = normalize_tensor(
            tensor,
            mode=mode,
            action_types=action_types[positions],
            reference=reference,
            reference_action_types=reference_types,
        )
        out[channel] = normed.reshape(normed.shape[0], -1).astype(np.float32)
    return out


def neighbor_sets(features: Mapping[str, np.ndarray], *, k: int, batch_size: int) -> dict[str, np.ndarray]:
    return {channel: cosine_knn_indices(x, k=k, batch_size=batch_size) for channel, x in features.items()}


def per_query_scores(a: np.ndarray, b: np.ndarray, *, k: int) -> np.ndarray:
    stats = []
    kk = min(int(k), a.shape[1], b.shape[1])
    for i in range(a.shape[0]):
        left = [int(v) for v in a[i, :kk].tolist() if int(v) >= 0]
        right = [int(v) for v in b[i, :kk].tolist() if int(v) >= 0]
        denom = min(kk, len(left), len(right))
        stats.append(len(set(left) & set(right)) / float(denom) if denom else float("nan"))
    return np.asarray(stats, dtype=np.float32)


def per_query_cross_scores(pa: np.ndarray, tb: np.ndarray, pb: np.ndarray, ta: np.ndarray, *, k: int) -> np.ndarray:
    ab = per_query_scores(pa, tb, k=k)
    ba = per_query_scores(pb, ta, k=k)
    return np.nanmean(np.stack([ab, ba], axis=0), axis=0).astype(np.float32)


def score_maps(
    data: Mapping[str, np.ndarray],
    channels: Sequence[str],
    full_rows: np.ndarray,
    probe_pos: np.ndarray,
    fraction_rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    n_states, _ = full_rows.shape
    pair_scores: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    fraction_by_pair: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for i, a in enumerate(channels):
        for b in channels[i + 1 :]:
            fraction_by_pair[(a, b)] = {}
    for row in fraction_rows:
        key = (str(row["channel_a"]), str(row["channel_b"]))
        state_row = int(row["state_row"])
        if key not in fraction_by_pair:
            continue
        for metric in ["regular_both_fraction", "blind_either_fraction", "physical_nonnull_fraction"]:
            fraction_by_pair[key].setdefault(metric, np.zeros(n_states, dtype=np.float32))[state_row] = float(row[metric])

    world = np.asarray(data["world_delta"], dtype=np.float32)[full_rows][:, probe_pos]
    for i, a in enumerate(channels):
        da = np.asarray(data[f"detect_{a}"], dtype=np.float32)[full_rows][:, probe_pos]
        da_rank = rank01(da.reshape(-1)).reshape(da.shape)
        for b in channels[i + 1 :]:
            db = np.asarray(data[f"detect_{b}"], dtype=np.float32)[full_rows][:, probe_pos]
            db_rank = rank01(db.reshape(-1)).reshape(db.shape)
            frac = fraction_by_pair[(a, b)]
            regular = frac["regular_both_fraction"]
            blind = frac["blind_either_fraction"]
            pair_scores[(a, b)] = {
                "regular_minus_blind": regular - blind,
                "regular_both_fraction": regular,
                "blind_either_fraction": blind,
                "physical_nonnull_fraction": frac["physical_nonnull_fraction"],
                "detect_geom_raw_mean": np.mean(np.sqrt(np.maximum(da, 0) * np.maximum(db, 0)), axis=1),
                "detect_geom_rank_mean": np.mean(np.sqrt(np.maximum(da_rank, 0) * np.maximum(db_rank, 0)), axis=1),
                "mean_world_delta": np.mean(world, axis=1),
            }
    return pair_scores


def correlation_rows(
    score_by_pair: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    per_query_by_control: Mapping[str, Mapping[tuple[str, str], np.ndarray]],
) -> list[dict[str, object]]:
    rows = []
    for control, pair_scores in per_query_by_control.items():
        for pair, overlap_scores in pair_scores.items():
            for score_name, score_values in score_by_pair[pair].items():
                pearson, spearman, n = corr(score_values, overlap_scores)
                rows.append(
                    {
                        "control": control,
                        "channel_a": pair[0],
                        "channel_b": pair[1],
                        "score": score_name,
                        "n": n,
                        "pearson": pearson,
                        "spearman": spearman,
                    }
                )
    return rows


def quantile_rows(
    score_by_pair: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    per_query_by_control: Mapping[str, Mapping[tuple[str, str], np.ndarray]],
    *,
    bins: int,
) -> list[dict[str, object]]:
    rows = []
    for control, pair_scores in per_query_by_control.items():
        for pair, overlap_scores in pair_scores.items():
            for score_name, score_values in score_by_pair[pair].items():
                score_values = np.asarray(score_values, dtype=np.float32)
                overlap_scores = np.asarray(overlap_scores, dtype=np.float32)
                mask = np.isfinite(score_values) & np.isfinite(overlap_scores)
                if int(mask.sum()) < bins:
                    continue
                order = np.argsort(score_values[mask], kind="mergesort")
                idx = np.nonzero(mask)[0][order]
                for bin_id, chunk in enumerate(np.array_split(idx, int(bins))):
                    if chunk.size == 0:
                        continue
                    rows.append(
                        {
                            "control": control,
                            "channel_a": pair[0],
                            "channel_b": pair[1],
                            "score": score_name,
                            "quantile_bin": int(bin_id),
                            "n": int(chunk.size),
                            "score_min": float(np.min(score_values[chunk])),
                            "score_mean": float(np.mean(score_values[chunk])),
                            "score_max": float(np.max(score_values[chunk])),
                            "overlap_mean": float(np.mean(overlap_scores[chunk])),
                        }
                    )
    return rows


def gate_summary_rows(overlap_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_key = {}
    for row in overlap_rows:
        if str(row["stratum"]) != "all":
            continue
        key = (str(row["normalization"]), str(row["channel_a"]), str(row["channel_b"]), str(row["control"]))
        by_key[key] = row
    rows = []
    for norm in sorted({str(row["normalization"]) for row in overlap_rows}):
        for pair in [("rgb", "noisy_rgb"), ("rgb", "gray_rgb"), ("rgb", "blur_rgb"), ("rgb", "range")]:
            cross = by_key.get((norm, pair[0], pair[1], "probe_to_heldout_cross"))
            held = by_key.get((norm, pair[0], pair[1], "action_effect_heldout_signature"))
            static = by_key.get((norm, pair[0], pair[1], "static_heldout_signature"))
            shuffled = by_key.get((norm, pair[0], pair[1], "action_column_shuffled_heldout"))
            if not (cross and held and static and shuffled):
                continue
            cross_adj = float(cross["chance_adjusted_overlap"])
            held_adj = float(held["chance_adjusted_overlap"])
            static_adj = float(static["chance_adjusted_overlap"])
            shuffled_adj = float(shuffled["chance_adjusted_overlap"])
            rows.append(
                {
                    "normalization": norm,
                    "normalization_primary": bool(normalization_is_primary(norm)),
                    "normalization_transductive": bool(normalization_is_transductive(norm)),
                    "channel_a": pair[0],
                    "channel_b": pair[1],
                    "gate0_probe_to_heldout_positive": bool(cross_adj > 0.0),
                    "gate1_action_effect_gt_static": bool(held_adj > static_adj),
                    "gate1_action_effect_gt_shuffled": bool(held_adj > shuffled_adj),
                    "probe_to_heldout_adjusted": cross_adj,
                    "heldout_adjusted": held_adj,
                    "static_adjusted": static_adj,
                    "shuffled_heldout_adjusted": shuffled_adj,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    data = load_npz(args.data)
    encoders, metadata = load_transition_encoders(args.model)
    static_encoders, static_metadata = load_transition_encoders(args.static_model)
    validate_model_data_path(metadata, args.data, allow_cross_data_model=bool(args.allow_cross_data_model))
    validate_model_data_path(static_metadata, args.data, allow_cross_data_model=bool(args.allow_cross_data_model))
    channels = [str(x) for x in (args.channels or list(encoders.keys()))]
    leakage = sorted(set(channels) & diagnostic_only_channels(data))
    if leakage:
        raise ValueError(f"Stage B.3 primary gates exclude diagnostic-only channels: {leakage}")
    sample_indices = validate_dense_stage0_data(data, channels, require_detect=True)
    static_sample_indices = validate_dense_static_data(data, channels)
    validate_model_fingerprint(metadata, data, channels, sample_indices, prefix="delta", allow_cross_data_model=bool(args.allow_cross_data_model))
    validate_model_fingerprint(static_metadata, data, channels, static_sample_indices, prefix="obs0", allow_cross_data_model=bool(args.allow_cross_data_model))
    state_ids, action_ids, dense_rows, full_rows = complete_state_action_matrix(
        data, sample_indices, max_states=int(args.max_states), seed=int(args.seed)
    )
    probe_set = {int(x) for x in args.probe_action_ids}
    probe_pos = np.asarray([i for i, action_id in enumerate(action_ids.tolist()) if int(action_id) in probe_set], dtype=np.int64)
    test_pos = np.asarray([i for i, action_id in enumerate(action_ids.tolist()) if int(action_id) not in probe_set], dtype=np.int64)
    if probe_pos.size == 0 or test_pos.size == 0:
        raise ValueError("Probe split must select at least one action and leave held-out actions.")
    full_action_types = np.asarray(data["action_type"]).astype(str)[full_rows[0]]
    missing_probe_types = sorted(set(full_action_types[test_pos].tolist()) - set(full_action_types[probe_pos].tolist()))
    if missing_probe_types:
        raise ValueError(
            "Held-out actions contain action types absent from probe actions; "
            f"probe_action_type_apply would be ill-defined: {missing_probe_types}"
        )
    validate_probe_trained_model(
        metadata,
        [int(action_ids[i]) for i in probe_pos.tolist()],
        allow_all_action_trained_model=bool(args.allow_all_action_trained_model),
    )
    validate_probe_trained_model(
        static_metadata,
        [int(action_ids[i]) for i in probe_pos.tolist()],
        allow_all_action_trained_model=bool(args.allow_all_action_trained_model),
    )
    pair_strata, fraction_rows, thresholds = state_pair_strata(
        data,
        channels,
        full_rows,
        probe_pos,
        detectability_quantile=float(args.detectability_quantile),
        regular_threshold=float(args.regular_state_threshold),
        blind_threshold=float(args.blind_state_threshold),
        physical_threshold=float(args.physical_state_threshold),
    )
    static_dense_rows = dense_rows_for_sample_ids(static_sample_indices, full_rows)
    action_tensors = embedding_tensors(data, encoders, channels, prefix="delta", dense_rows=dense_rows)
    static_tensors = embedding_tensors(data, static_encoders, channels, prefix="obs0", dense_rows=static_dense_rows)
    score_by_pair = score_maps(data, channels, full_rows, probe_pos, fraction_rows)

    all_overlap_rows: list[dict[str, object]] = []
    all_bootstrap_rows: list[dict[str, object]] = []
    all_corr_rows: list[dict[str, object]] = []
    all_quantile_rows: list[dict[str, object]] = []

    for norm in [str(x) for x in args.normalization_modes]:
        if norm not in NORMALIZATION_MODES:
            raise ValueError(f"Unsupported normalization mode: {norm}")
        ref_pos = probe_pos if norm in {"probe_global_apply", "probe_action_type_apply"} else None
        probe_features = features_from_tensors(
            action_tensors,
            channels,
            probe_pos,
            mode=norm,
            action_types=full_action_types,
            reference_positions=ref_pos,
        )
        test_features = features_from_tensors(
            action_tensors,
            channels,
            test_pos,
            mode=norm,
            action_types=full_action_types,
            reference_positions=ref_pos,
        )
        static_features = features_from_tensors(
            static_tensors,
            channels,
            test_pos,
            mode=norm,
            action_types=full_action_types,
            reference_positions=ref_pos,
        )
        shuffled_probe_features = shuffled_features_from_tensors(
            action_tensors,
            channels,
            probe_pos,
            mode=norm,
            action_types=full_action_types,
            reference_positions=ref_pos,
            seed=int(args.seed) + 1003,
        )
        shuffled_heldout_features = shuffled_features_from_tensors(
            action_tensors,
            channels,
            test_pos,
            mode=norm,
            action_types=full_action_types,
            reference_positions=ref_pos,
            seed=int(args.seed) + 1004,
        )
        probe_neighbors = neighbor_sets(probe_features, k=int(args.k), batch_size=int(args.batch_size))
        test_neighbors = neighbor_sets(test_features, k=int(args.k), batch_size=int(args.batch_size))
        static_neighbors = neighbor_sets(static_features, k=int(args.k), batch_size=int(args.batch_size))
        shuffled_probe_neighbors = neighbor_sets(shuffled_probe_features, k=int(args.k), batch_size=int(args.batch_size))
        shuffled_heldout_neighbors = neighbor_sets(
            shuffled_heldout_features, k=int(args.k), batch_size=int(args.batch_size)
        )

        rows = []
        for control, neighbors in [
            ("action_effect_probe_signature", probe_neighbors),
            ("action_effect_heldout_signature", test_neighbors),
            ("static_heldout_signature", static_neighbors),
            ("action_column_shuffled_probe", shuffled_probe_neighbors),
            ("action_column_shuffled_heldout", shuffled_heldout_neighbors),
        ]:
            rows.extend(pairwise_stratified_overlap_rows(neighbors, pair_strata, k=int(args.k), control=control))
        rows.extend(cross_split_rows(probe_neighbors, test_neighbors, pair_strata, k=int(args.k), control="probe_to_heldout_cross"))
        for row in rows:
            row["normalization"] = norm
            row["normalization_primary"] = bool(normalization_is_primary(norm))
            row["normalization_transductive"] = bool(normalization_is_transductive(norm))
        all_overlap_rows.extend(rows)

        boot = bootstrap_rows(
            {
                "action_effect_probe_signature": probe_neighbors,
                "action_effect_heldout_signature": test_neighbors,
                "static_heldout_signature": static_neighbors,
                "action_column_shuffled_probe": shuffled_probe_neighbors,
                "action_column_shuffled_heldout": shuffled_heldout_neighbors,
            },
            pair_strata,
            k=int(args.k),
            repeats=int(args.bootstrap_repeats),
            seed=int(args.bootstrap_seed) + 101,
        )
        boot.extend(
            bootstrap_cross_rows(
                probe_neighbors,
                test_neighbors,
                pair_strata,
                k=int(args.k),
                repeats=int(args.bootstrap_repeats),
                seed=int(args.bootstrap_seed) + 102,
                control="probe_to_heldout_cross",
            )
        )
        for row in boot:
            row["normalization"] = norm
            row["normalization_primary"] = bool(normalization_is_primary(norm))
            row["normalization_transductive"] = bool(normalization_is_transductive(norm))
        all_bootstrap_rows.extend(boot)

        per_query_by_control: dict[str, dict[tuple[str, str], np.ndarray]] = {
            "action_effect_heldout_signature": {},
            "static_heldout_signature": {},
            "action_column_shuffled_probe": {},
            "action_column_shuffled_heldout": {},
            "probe_to_heldout_cross": {},
        }
        for i, a in enumerate(channels):
            for b in channels[i + 1 :]:
                pair = (a, b)
                per_query_by_control["action_effect_heldout_signature"][pair] = per_query_scores(
                    test_neighbors[a], test_neighbors[b], k=int(args.k)
                )
                per_query_by_control["static_heldout_signature"][pair] = per_query_scores(
                    static_neighbors[a], static_neighbors[b], k=int(args.k)
                )
                per_query_by_control["action_column_shuffled_probe"][pair] = per_query_scores(
                    shuffled_probe_neighbors[a], shuffled_probe_neighbors[b], k=int(args.k)
                )
                per_query_by_control["action_column_shuffled_heldout"][pair] = per_query_scores(
                    shuffled_heldout_neighbors[a], shuffled_heldout_neighbors[b], k=int(args.k)
                )
                per_query_by_control["probe_to_heldout_cross"][pair] = per_query_cross_scores(
                    probe_neighbors[a],
                    test_neighbors[b],
                    probe_neighbors[b],
                    test_neighbors[a],
                    k=int(args.k),
                )
        corr_rows = correlation_rows(score_by_pair, per_query_by_control)
        quant_rows = quantile_rows(score_by_pair, per_query_by_control, bins=int(args.score_quantile_bins))
        for row in corr_rows:
            row["normalization"] = norm
            row["normalization_primary"] = bool(normalization_is_primary(norm))
            row["normalization_transductive"] = bool(normalization_is_transductive(norm))
        for row in quant_rows:
            row["normalization"] = norm
            row["normalization_primary"] = bool(normalization_is_primary(norm))
            row["normalization_transductive"] = bool(normalization_is_transductive(norm))
        all_corr_rows.extend(corr_rows)
        all_quantile_rows.extend(quant_rows)

    out = ensure_dir(args.out)
    reports = ensure_dir(out / "reports")
    write_csv(reports / "normalization_sweep_knn.csv", all_overlap_rows)
    write_csv(reports / "normalization_sweep_bootstrap_ci.csv", all_bootstrap_rows)
    write_csv(reports / "observability_score_vs_overlap.csv", all_corr_rows)
    write_csv(
        reports / "regular_minus_blind_correlation.csv",
        [row for row in all_corr_rows if row["score"] == "regular_minus_blind"],
    )
    write_csv(reports / "score_quantile_knn.csv", all_quantile_rows)
    write_csv(reports / "state_strata_fraction_summary.csv", summarize_fraction_rows(fraction_rows))
    write_csv(reports / "gate_summary.csv", gate_summary_rows(all_overlap_rows))
    save_json(
        {
            "data": str(Path(args.data)),
            "model": str(Path(args.model)),
            "static_model": str(Path(args.static_model)),
            "channels": channels,
            "normalization_modes": [str(x) for x in args.normalization_modes],
            "primary_normalization_modes": sorted(PRIMARY_NORMALIZATION_MODES),
            "k": int(args.k),
            "n_states": int(state_ids.shape[0]),
            "n_actions": int(action_ids.shape[0]),
            "probe_action_ids": [int(action_ids[i]) for i in probe_pos.tolist()],
            "heldout_action_ids": [int(action_ids[i]) for i in test_pos.tolist()],
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "score_quantile_bins": int(args.score_quantile_bins),
            "strata_thresholds": thresholds,
            "decision_table": {
                "gate0_fails": "repair metric / normalization before interpreting rgb-range",
                "gate0_gate1_pass_gate2_continuous": "Option 2.5 continuous action-effect observability",
                "gate0_gate1_pass_gate2_fails": "Option 3-lite action as universal coupling candidate",
                "gate1_fails": "environment / encoder redesign",
            },
        },
        reports / "stageb3_summary.json",
    )
    print(f"Wrote Stage B.3 gate reports to {out}")


if __name__ == "__main__":
    main()
