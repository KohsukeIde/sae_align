#!/usr/bin/env python
"""Stage B.5 held-out same-action-set cross-channel alignment."""

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
from typing import Mapping, Sequence

import numpy as np

from analyze_stageb3_gates import (
    NORMALIZATION_MODES,
    PRIMARY_NORMALIZATION_MODES,
    corr,
    features_from_tensors,
    normalization_is_primary,
    normalization_is_transductive,
    rank01,
    score_maps,
    shuffled_features_from_tensors,
)
from analyze_stageb4_reliability import (
    DEFAULT_REDUNDANCY_PAIRS,
    CORE_REDUNDANCY_PAIRS,
    adjusted_unit,
    bootstrap_paired_diff_ci,
    feature_diagnostic_rows,
)
from analyze_state_signature_knn import (
    bootstrap_mean_ci,
    complete_state_action_matrix,
    dense_rows_for_sample_ids,
    load_npz,
    state_pair_strata,
    validate_model_data_path,
    validate_model_fingerprint,
    validate_probe_trained_model,
)
from sae_align.analysis.knn_alignment import cosine_knn_indices
from sae_align.analysis.strata import diagnostic_only_channels, validate_dense_stage0_data, validate_dense_static_data
from sae_align.models import load_transition_encoders
from sae_align.utils.io import ensure_dir, save_json


DEFAULT_CHANNELS = ("rgb", "range", "local", "noisy_rgb", "gray_rgb", "blur_rgb")
TARGET_PAIRS = (("rgb", "range"), ("rgb", "local"), ("range", "local"))
B2_RGB_RANGE_ADJUSTED = 0.0400


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model", type=str, required=True, help="Probe-action-only action-effect PCA model.")
    p.add_argument("--static-model", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--all-action-model", type=str, default=None)
    p.add_argument("--channels", nargs="*", default=list(DEFAULT_CHANNELS))
    p.add_argument("--probe-action-ids", nargs="*", type=int, required=True)
    p.add_argument("--representations", nargs="*", default=["pca_probe_only", "raw_delta", "random_projection"])
    p.add_argument("--random-projection-dim", type=int, default=32)
    p.add_argument("--random-projection-seed", type=int, default=0)
    p.add_argument("--normalization-modes", nargs="*", default=["none", "probe_global_apply", "probe_action_type_apply"])
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
    p.add_argument("--allow-cross-data-model", action="store_true")
    return p.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def chance_level(neighbors: np.ndarray, *, k: int) -> tuple[int, float]:
    n = int(neighbors.shape[0])
    kk = min(int(k), int(neighbors.shape[1]))
    chance = float(kk / max(1, n - 1)) if n > 1 else float("nan")
    return kk, chance


def per_query_scores(a: np.ndarray, b: np.ndarray, *, k: int) -> np.ndarray:
    kk = min(int(k), a.shape[1], b.shape[1])
    scores = []
    for i in range(a.shape[0]):
        left = [int(v) for v in a[i, :kk].tolist() if int(v) >= 0]
        right = [int(v) for v in b[i, :kk].tolist() if int(v) >= 0]
        denom = min(kk, len(left), len(right))
        scores.append(len(set(left) & set(right)) / float(denom) if denom else float("nan"))
    return np.asarray(scores, dtype=np.float32)


def neighbor_sets(features: Mapping[str, np.ndarray], *, k: int, batch_size: int) -> dict[str, np.ndarray]:
    return {channel: cosine_knn_indices(x, k=k, batch_size=batch_size) for channel, x in features.items()}


def embedding_tensors_from_model(
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


def raw_tensors(
    data: Mapping[str, np.ndarray],
    channels: Sequence[str],
    *,
    prefix: str,
    dense_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    n_states, n_actions = dense_rows.shape
    flat = dense_rows.reshape(-1)
    out = {}
    for channel in channels:
        x = np.asarray(data[f"{prefix}_{channel}"][flat], dtype=np.float32)
        out[channel] = x.reshape(n_states, n_actions, -1)
    return out


def random_project_tensors(
    tensors: Mapping[str, np.ndarray],
    *,
    dim: int,
    seed: int,
) -> dict[str, np.ndarray]:
    out = {}
    for idx, (channel, tensor) in enumerate(tensors.items()):
        flat = tensor.reshape(-1, tensor.shape[-1])
        rng = np.random.default_rng(int(seed) + idx * 1009)
        proj = rng.normal(0.0, 1.0 / np.sqrt(max(1, flat.shape[1])), size=(flat.shape[1], int(dim))).astype(np.float32)
        y = (flat @ proj).astype(np.float32)
        out[channel] = y.reshape(tensor.shape[0], tensor.shape[1], int(dim))
    return out


def overlap_row(
    *,
    representation: str,
    control: str,
    channel_a: str,
    channel_b: str,
    normalization: str,
    neighbors: Mapping[str, np.ndarray],
    k: int,
) -> tuple[dict[str, object], np.ndarray]:
    scores = per_query_scores(neighbors[channel_a], neighbors[channel_b], k=int(k))
    kk, chance = chance_level(neighbors[channel_a], k=int(k))
    overlap = float(np.nanmean(scores)) if scores.size else float("nan")
    row = {
        "representation": representation,
        "representation_primary": bool(representation == "pca_probe_only"),
        "representation_diagnostic": bool(representation != "pca_probe_only"),
        "control": control,
        "channel_a": channel_a,
        "channel_b": channel_b,
        "normalization": normalization,
        "normalization_primary": bool(normalization_is_primary(normalization)),
        "normalization_transductive": bool(normalization_is_transductive(normalization)),
        "n_queries": int(neighbors[channel_a].shape[0]),
        "k": int(k),
        "n_valid_queries": int(np.isfinite(scores).sum()),
        "mean_effective_k": float(kk),
        "overlap": overlap,
        "random_expected_overlap": chance,
        "chance_adjusted_overlap": overlap - chance if np.isfinite(overlap) else float("nan"),
        "chance_adjusted_unit_overlap": adjusted_unit(overlap, chance),
    }
    return row, scores


def bootstrap_row(base: Mapping[str, object], scores: np.ndarray, *, repeats: int, seed: int) -> dict[str, object]:
    clean = scores[np.isfinite(scores)]
    mean, low, high = bootstrap_mean_ci(clean, repeats=int(repeats), seed=int(seed))
    chance = float(base["random_expected_overlap"])
    return {
        **base,
        "overlap_mean": mean,
        "overlap_ci95_low": low,
        "overlap_ci95_high": high,
        "chance_adjusted_mean": mean - chance if np.isfinite(mean) else float("nan"),
        "chance_adjusted_ci95_low": low - chance if np.isfinite(low) else float("nan"),
        "chance_adjusted_ci95_high": high - chance if np.isfinite(high) else float("nan"),
        "bootstrap_repeats": int(repeats),
    }


def b2_branch(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if value >= B2_RGB_RANGE_ADJUSTED + 0.0200:
        return "strengthened"
    if value >= 0.0200:
        return "replicated_weak_positive"
    if value > 0.0:
        return "diminished"
    return "disappeared"


def mean_finite(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def corr_rows_for_scores(
    *,
    representation: str,
    control: str,
    normalization: str,
    pair_scores: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    per_query_by_pair: Mapping[tuple[str, str], np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    for pair, query_scores in per_query_by_pair.items():
        for score_name, score_values in pair_scores[pair].items():
            pearson, spearman, n = corr(score_values, query_scores)
            rows.append(
                {
                    "representation": representation,
                    "control": control,
                    "normalization": normalization,
                    "channel_a": pair[0],
                    "channel_b": pair[1],
                    "score": score_name,
                    "score_primary": bool(score_name == "detect_geom_rank_mean"),
                    "n": n,
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )
    return rows


def quantile_rows_for_scores(
    *,
    representation: str,
    control: str,
    normalization: str,
    pair_scores: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    per_query_by_pair: Mapping[tuple[str, str], np.ndarray],
    bins: int,
) -> list[dict[str, object]]:
    rows = []
    for pair, query_scores in per_query_by_pair.items():
        query_scores = np.asarray(query_scores, dtype=np.float32)
        for score_name, score_values in pair_scores[pair].items():
            score_values = np.asarray(score_values, dtype=np.float32)
            mask = np.isfinite(score_values) & np.isfinite(query_scores)
            if int(mask.sum()) < int(bins):
                continue
            idx = np.nonzero(mask)[0][np.argsort(score_values[mask], kind="mergesort")]
            for bin_id, chunk in enumerate(np.array_split(idx, int(bins))):
                if chunk.size == 0:
                    continue
                rows.append(
                    {
                        "representation": representation,
                        "control": control,
                        "normalization": normalization,
                        "channel_a": pair[0],
                        "channel_b": pair[1],
                        "score": score_name,
                        "score_primary": bool(score_name == "detect_geom_rank_mean"),
                        "quantile_bin": int(bin_id),
                        "n": int(chunk.size),
                        "score_mean": float(np.mean(score_values[chunk])),
                        "overlap_mean": float(np.mean(query_scores[chunk])),
                    }
                )
    return rows


def gate_rows(
    overlap_rows: Sequence[Mapping[str, object]],
    bootstrap_rows: Sequence[Mapping[str, object]],
    diff_rows: Sequence[Mapping[str, object]],
    corr_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_key = {
        (str(r["representation"]), str(r["normalization"]), str(r["control"]), str(r["channel_a"]), str(r["channel_b"])): r
        for r in overlap_rows
    }
    diff_by_key = {
        (
            str(r["representation"]),
            str(r["normalization"]),
            str(r["channel_a"]),
            str(r["channel_b"]),
            str(r["comparison"]),
        ): r
        for r in diff_rows
    }
    boot_by_key = {
        (str(r["representation"]), str(r["normalization"]), str(r["control"]), str(r["channel_a"]), str(r["channel_b"])): r
        for r in bootstrap_rows
    }
    corr_by_key = {
        (str(r["representation"]), str(r["normalization"]), str(r["channel_a"]), str(r["channel_b"]), str(r["score"])): r
        for r in corr_rows
        if str(r["control"]) == "action_effect_heldout_signature"
    }
    rows = []
    for rep in sorted({str(r["representation"]) for r in overlap_rows}):
        for norm in sorted({str(r["normalization"]) for r in overlap_rows if str(r["representation"]) == rep}):
            core_passes = []
            core_ci_passes = []
            core_shuffled_ci_passes = []
            for a, b in CORE_REDUNDANCY_PAIRS:
                row = by_key.get((rep, norm, "action_effect_heldout_signature", a, b))
                core_passes.append(bool(row and float(row["chance_adjusted_overlap"]) > 0.0))
                boot = boot_by_key.get((rep, norm, "action_effect_heldout_signature", a, b))
                core_ci_passes.append(bool(boot and float(boot["chance_adjusted_ci95_low"]) > 0.0))
                diff = diff_by_key.get((rep, norm, a, b, "action_effect_minus_shuffled"))
                core_shuffled_ci_passes.append(bool(diff and float(diff["diff_ci95_low"]) > 0.0))
            rgb_range = by_key.get((rep, norm, "action_effect_heldout_signature", "rgb", "range"))
            rgb_range_boot = boot_by_key.get((rep, norm, "action_effect_heldout_signature", "rgb", "range"))
            ae_static = diff_by_key.get((rep, norm, "rgb", "range", "action_effect_minus_static"))
            ae_shuffled = diff_by_key.get((rep, norm, "rgb", "range", "action_effect_minus_shuffled"))
            obs = corr_by_key.get((rep, norm, "rgb", "range", "detect_geom_rank_mean"))
            rgb_range_adjusted = float(rgb_range["chance_adjusted_overlap"]) if rgb_range else float("nan")
            rows.append(
                {
                    "representation": rep,
                    "normalization": norm,
                    "normalization_primary": bool(normalization_is_primary(norm)),
                    "gate0_core_redundancy_point_pass": bool(core_passes and all(core_passes)),
                    "gate0_core_redundancy_ci_pass": bool(core_ci_passes and all(core_ci_passes)),
                    "gate0_core_redundancy_gt_shuffled_ci_pass": bool(
                        core_shuffled_ci_passes and all(core_shuffled_ci_passes)
                    ),
                    "gate1_rgb_range_adjusted": rgb_range_adjusted,
                    "gate1_rgb_range_ci_low": float(rgb_range_boot["chance_adjusted_ci95_low"])
                    if rgb_range_boot
                    else float("nan"),
                    "gate1_rgb_range_ci_pass": bool(
                        rgb_range_boot and float(rgb_range_boot["chance_adjusted_ci95_low"]) > 0.0
                    ),
                    "gate1_b2_branch": b2_branch(rgb_range_adjusted),
                    "gate1_action_effect_gt_static_ci_pass": bool(
                        ae_static and float(ae_static["diff_ci95_low"]) > 0.0
                    ),
                    "gate1_action_effect_gt_shuffled_ci_pass": bool(
                        ae_shuffled and float(ae_shuffled["diff_ci95_low"]) > 0.0
                    ),
                    "gate2_detect_geom_rank_spearman": float(obs["spearman"]) if obs else float("nan"),
                    "stagec_candidate": bool(
                        core_passes
                        and all(core_passes)
                        and core_ci_passes
                        and all(core_ci_passes)
                        and core_shuffled_ci_passes
                        and all(core_shuffled_ci_passes)
                        and rgb_range_boot
                        and float(rgb_range_boot["chance_adjusted_ci95_low"]) > 0.0
                        and ae_static
                        and float(ae_static["diff_ci95_low"]) > 0.0
                        and ae_shuffled
                        and float(ae_shuffled["diff_ci95_low"]) > 0.0
                    ),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    data = load_npz(args.data)
    channels = [str(x) for x in args.channels]
    leakage = sorted(set(channels) & diagnostic_only_channels(data))
    if leakage:
        raise ValueError(f"Stage B.5 excludes diagnostic-only channels: {leakage}")
    sample_indices = validate_dense_stage0_data(data, channels, require_detect=True)
    static_sample_indices = validate_dense_static_data(data, channels)
    state_ids, action_ids, dense_rows, full_rows = complete_state_action_matrix(
        data, sample_indices, max_states=int(args.max_states), seed=int(args.seed)
    )
    probe_set = {int(x) for x in args.probe_action_ids}
    probe_pos = np.asarray([i for i, action_id in enumerate(action_ids.tolist()) if int(action_id) in probe_set], dtype=np.int64)
    heldout_pos = np.asarray([i for i, action_id in enumerate(action_ids.tolist()) if int(action_id) not in probe_set], dtype=np.int64)
    if probe_pos.size == 0 or heldout_pos.size == 0:
        raise ValueError("Probe split must select at least one action and leave held-out actions.")
    full_action_types = np.asarray(data["action_type"]).astype(str)[full_rows[0]]
    missing_probe_types = sorted(set(full_action_types[heldout_pos].tolist()) - set(full_action_types[probe_pos].tolist()))
    if missing_probe_types:
        raise ValueError(
            "Held-out actions contain action types absent from probe actions; "
            f"probe_action_type_apply would be ill-defined: {missing_probe_types}"
        )

    encoders, metadata = load_transition_encoders(args.model)
    static_encoders, static_metadata = load_transition_encoders(args.static_model)
    validate_model_data_path(metadata, args.data, allow_cross_data_model=bool(args.allow_cross_data_model))
    validate_model_data_path(static_metadata, args.data, allow_cross_data_model=bool(args.allow_cross_data_model))
    validate_model_fingerprint(metadata, data, channels, sample_indices, prefix="delta", allow_cross_data_model=bool(args.allow_cross_data_model))
    validate_model_fingerprint(static_metadata, data, channels, static_sample_indices, prefix="obs0", allow_cross_data_model=bool(args.allow_cross_data_model))
    validate_probe_trained_model(metadata, [int(action_ids[i]) for i in probe_pos.tolist()], allow_all_action_trained_model=False)
    validate_probe_trained_model(
        static_metadata,
        [int(action_ids[i]) for i in probe_pos.tolist()],
        allow_all_action_trained_model=False,
    )

    representation_tensors: dict[str, dict[str, np.ndarray]] = {}
    if "pca_probe_only" in args.representations:
        representation_tensors["pca_probe_only"] = embedding_tensors_from_model(
            data, encoders, channels, prefix="delta", dense_rows=dense_rows
        )
    raw = raw_tensors(data, channels, prefix="delta", dense_rows=dense_rows)
    if "raw_delta" in args.representations:
        representation_tensors["raw_delta"] = raw
    if "random_projection" in args.representations:
        representation_tensors["random_projection"] = random_project_tensors(
            raw,
            dim=int(args.random_projection_dim),
            seed=int(args.random_projection_seed),
        )
    if "pca_all_action" in args.representations:
        if not args.all_action_model:
            raise ValueError("representation pca_all_action requires --all-action-model")
        all_encoders, all_metadata = load_transition_encoders(args.all_action_model)
        validate_model_data_path(all_metadata, args.data, allow_cross_data_model=bool(args.allow_cross_data_model))
        validate_model_fingerprint(
            all_metadata,
            data,
            channels,
            sample_indices,
            prefix="delta",
            allow_cross_data_model=bool(args.allow_cross_data_model),
        )
        representation_tensors["pca_all_action"] = embedding_tensors_from_model(
            data, all_encoders, channels, prefix="delta", dense_rows=dense_rows
        )

    static_dense_rows = dense_rows_for_sample_ids(static_sample_indices, full_rows)
    static_tensors = embedding_tensors_from_model(
        data, static_encoders, channels, prefix="obs0", dense_rows=static_dense_rows
    )
    pair_strata, fraction_rows, thresholds = state_pair_strata(
        data,
        channels,
        full_rows,
        heldout_pos,
        detectability_quantile=float(args.detectability_quantile),
        regular_threshold=float(args.regular_state_threshold),
        blind_threshold=float(args.blind_state_threshold),
        physical_threshold=float(args.physical_state_threshold),
    )
    score_by_pair = score_maps(data, channels, full_rows, heldout_pos, fraction_rows)
    pairs = list(DEFAULT_REDUNDANCY_PAIRS) + list(TARGET_PAIRS)
    pairs = list(dict.fromkeys(pairs))
    rng = np.random.default_rng(int(args.bootstrap_seed))
    overlap_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    diff_rows: list[dict[str, object]] = []
    corr_rows: list[dict[str, object]] = []
    quantile_rows: list[dict[str, object]] = []
    tie_rows: list[dict[str, object]] = []

    for norm in [str(x) for x in args.normalization_modes]:
        if norm not in NORMALIZATION_MODES:
            raise ValueError(f"Unsupported normalization mode: {norm}")
        ref_pos = probe_pos if norm in {"probe_global_apply", "probe_action_type_apply"} else None
        static_features = features_from_tensors(
            static_tensors,
            channels,
            heldout_pos,
            mode=norm,
            action_types=full_action_types,
            reference_positions=ref_pos,
        )
        static_neighbors = neighbor_sets(static_features, k=int(args.k), batch_size=int(args.batch_size))
        for rep, tensors in representation_tensors.items():
            features = features_from_tensors(
                tensors,
                channels,
                heldout_pos,
                mode=norm,
                action_types=full_action_types,
                reference_positions=ref_pos,
            )
            shuffled = shuffled_features_from_tensors(
                tensors,
                channels,
                heldout_pos,
                mode=norm,
                action_types=full_action_types,
                reference_positions=ref_pos,
                seed=int(args.seed) + 7001,
            )
            tie_rows.extend(
                {**row, "representation": rep}
                for row in feature_diagnostic_rows(features, feature_set="heldout", normalization=norm, k=int(args.k))
            )
            neighbors = neighbor_sets(features, k=int(args.k), batch_size=int(args.batch_size))
            shuffled_neighbors = neighbor_sets(shuffled, k=int(args.k), batch_size=int(args.batch_size))
            per_query_by_pair: dict[tuple[str, str], np.ndarray] = {}
            for control, current_neighbors in [
                ("action_effect_heldout_signature", neighbors),
                ("static_heldout_signature", static_neighbors),
                ("action_column_shuffled_heldout", shuffled_neighbors),
            ]:
                for a, b in pairs:
                    if a not in channels or b not in channels:
                        continue
                    base, scores = overlap_row(
                        representation=rep,
                        control=control,
                        channel_a=a,
                        channel_b=b,
                        normalization=norm,
                        neighbors=current_neighbors,
                        k=int(args.k),
                    )
                    overlap_rows.append(base)
                    bootstrap_rows.append(
                        bootstrap_row(
                            base,
                            scores,
                            repeats=int(args.bootstrap_repeats),
                            seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                        )
                    )
                    if control == "action_effect_heldout_signature":
                        per_query_by_pair[(a, b)] = scores
                if control == "action_effect_heldout_signature":
                    continue
            for a, b in pairs:
                if a not in channels or b not in channels:
                    continue
                ae = per_query_scores(neighbors[a], neighbors[b], k=int(args.k))
                st = per_query_scores(static_neighbors[a], static_neighbors[b], k=int(args.k))
                sh = per_query_scores(shuffled_neighbors[a], shuffled_neighbors[b], k=int(args.k))
                for name, control_scores in [
                    ("action_effect_minus_static", st),
                    ("action_effect_minus_shuffled", sh),
                ]:
                    mean, low, high, n = bootstrap_paired_diff_ci(
                        ae,
                        control_scores,
                        repeats=int(args.bootstrap_repeats),
                        seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                    )
                    diff_rows.append(
                        {
                            "representation": rep,
                            "representation_primary": bool(rep == "pca_probe_only"),
                            "representation_diagnostic": bool(rep != "pca_probe_only"),
                            "comparison": name,
                            "channel_a": a,
                            "channel_b": b,
                            "normalization": norm,
                            "normalization_primary": bool(normalization_is_primary(norm)),
                            "normalization_transductive": bool(normalization_is_transductive(norm)),
                            "k": int(args.k),
                            "n_states": int(state_ids.shape[0]),
                            "n_probe_actions": int(probe_pos.size),
                            "n_heldout_actions": int(heldout_pos.size),
                            "n_valid_queries": int(n),
                            "observed_mean": mean_finite(ae),
                            "control_mean": mean_finite(control_scores),
                            "diff_mean": mean,
                            "diff_ci95_low": low,
                            "diff_ci95_high": high,
                            "bootstrap_repeats": int(args.bootstrap_repeats),
                        }
                    )
            corr_rows.extend(
                corr_rows_for_scores(
                    representation=rep,
                    control="action_effect_heldout_signature",
                    normalization=norm,
                    pair_scores=score_by_pair,
                    per_query_by_pair=per_query_by_pair,
                )
            )
            quantile_rows.extend(
                quantile_rows_for_scores(
                    representation=rep,
                    control="action_effect_heldout_signature",
                    normalization=norm,
                    pair_scores=score_by_pair,
                    per_query_by_pair=per_query_by_pair,
                    bins=int(args.score_quantile_bins),
                )
            )

    gates = gate_rows(overlap_rows, bootstrap_rows, diff_rows, corr_rows)
    b2_rows = [
        {
            "representation": row["representation"],
            "normalization": row["normalization"],
            "b2_v1_rgb_range_adjusted": B2_RGB_RANGE_ADJUSTED,
            "b5_rgb_range_adjusted": row["gate1_rgb_range_adjusted"],
            "branch": row["gate1_b2_branch"],
        }
        for row in gates
        if row["representation"] == "pca_probe_only" and row["normalization_primary"]
    ]
    out = ensure_dir(args.out)
    reports = ensure_dir(out / "reports")
    write_csv(reports / "heldout_same_action_cross_channel_alignment.csv", overlap_rows)
    write_csv(reports / "heldout_same_action_bootstrap_ci.csv", bootstrap_rows)
    write_csv(
        reports / "heldout_same_action_redundancy_calibration.csv",
        [row for row in overlap_rows if (row["channel_a"], row["channel_b"]) in DEFAULT_REDUNDANCY_PAIRS],
    )
    write_csv(reports / "heldout_action_effect_vs_static.csv", diff_rows)
    write_csv(reports / "observability_score_correlation.csv", corr_rows)
    write_csv(reports / "observability_score_quantiles.csv", quantile_rows)
    write_csv(reports / "feature_tie_diagnostics.csv", tie_rows)
    write_csv(reports / "feature_extractor_diagnostic_raw_random_pca.csv", overlap_rows)
    write_csv(reports / "b2_signal_comparison.csv", b2_rows)
    write_csv(reports / "gate_summary.csv", gates)
    save_json(
        {
            "data": str(Path(args.data)),
            "model": str(Path(args.model)),
            "static_model": str(Path(args.static_model)),
            "all_action_model": str(Path(args.all_action_model)) if args.all_action_model else None,
            "channels": channels,
            "representations": sorted(representation_tensors),
            "normalization_modes": [str(x) for x in args.normalization_modes],
            "primary_normalization_modes": sorted(PRIMARY_NORMALIZATION_MODES),
            "k": int(args.k),
            "n_states": int(state_ids.shape[0]),
            "n_actions": int(action_ids.shape[0]),
            "probe_action_ids": [int(action_ids[i]) for i in probe_pos.tolist()],
            "heldout_action_ids": [int(action_ids[i]) for i in heldout_pos.tolist()],
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "strata_thresholds": thresholds,
            "b2_v1_rgb_range_adjusted": B2_RGB_RANGE_ADJUSTED,
            "gate_order": [
                "gate0_heldout_same_action_redundancy",
                "gate1_rgb_range_action_effect_gt_static_and_shuffled",
                "gate2_continuous_observability",
            ],
        },
        reports / "stageb5_summary.json",
    )
    print(f"Wrote Stage B.5 held-out alignment reports to {out}")


if __name__ == "__main__":
    main()
