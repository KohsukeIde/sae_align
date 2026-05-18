#!/usr/bin/env python
"""Stage B.4 split-half reliability gates.

Stage B.4 checks whether state-level action-effect signatures are reliable
across action splits before interpreting cross-channel alignment results.
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
from typing import Mapping, Sequence

import numpy as np

from analyze_stageb3_gates import (
    NORMALIZATION_MODES,
    PRIMARY_NORMALIZATION_MODES,
    embedding_tensors,
    features_from_tensors,
    neighbor_sets,
    normalization_is_primary,
    normalization_is_transductive,
    per_query_cross_scores,
    per_query_scores,
    shuffled_features_from_tensors,
)
from analyze_state_signature_knn import (
    bootstrap_mean_ci,
    complete_state_action_matrix,
    load_npz,
    validate_model_data_path,
    validate_model_fingerprint,
    validate_probe_trained_model,
)
from sae_align.analysis.strata import diagnostic_only_channels, validate_dense_stage0_data
from sae_align.models import load_transition_encoders
from sae_align.utils.io import ensure_dir, save_json


DEFAULT_CHANNELS = ("rgb", "range", "local", "noisy_rgb", "gray_rgb", "blur_rgb")
DEFAULT_REDUNDANCY_PAIRS = (("rgb", "noisy_rgb"), ("rgb", "gray_rgb"), ("rgb", "blur_rgb"))
CORE_REDUNDANCY_PAIRS = (("rgb", "noisy_rgb"), ("rgb", "gray_rgb"))
CORE_RELIABILITY_CHANNELS = ("rgb", "noisy_rgb", "gray_rgb")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--channels", nargs="*", default=list(DEFAULT_CHANNELS))
    p.add_argument("--probe-action-ids", nargs="*", type=int, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-states", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap-repeats", type=int, default=200)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=512)
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


def chance_level(neighbors: np.ndarray, *, k: int) -> tuple[int, float]:
    n = int(neighbors.shape[0])
    kk = min(int(k), int(neighbors.shape[1]))
    chance = float(kk / max(1, n - 1)) if n > 1 else float("nan")
    return kk, chance


def adjusted_unit(overlap: float, chance: float) -> float:
    if not (np.isfinite(overlap) and np.isfinite(chance)) or chance >= 1.0:
        return float("nan")
    return (float(overlap) - float(chance)) / (1.0 - float(chance))


def add_common_context(
    row: dict[str, object],
    *,
    k: int,
    n_states: int,
    n_probe_actions: int,
    n_heldout_actions: int,
) -> dict[str, object]:
    row["k"] = int(k)
    row["n_states"] = int(n_states)
    row["n_probe_actions"] = int(n_probe_actions)
    row["n_heldout_actions"] = int(n_heldout_actions)
    return row


def channel_score_row(
    *,
    comparison: str,
    channel: str,
    normalization: str,
    neighbors_a: np.ndarray,
    neighbors_b: np.ndarray,
    k: int,
) -> tuple[dict[str, object], np.ndarray]:
    scores = per_query_scores(neighbors_a, neighbors_b, k=int(k))
    kk, chance = chance_level(neighbors_a, k=int(k))
    overlap = float(np.nanmean(scores)) if scores.size else float("nan")
    row = {
        "comparison": comparison,
        "channel": channel,
        "normalization": normalization,
        "normalization_primary": bool(normalization_is_primary(normalization)),
        "normalization_transductive": bool(normalization_is_transductive(normalization)),
        "n_queries": int(neighbors_a.shape[0]),
        "n_valid_queries": int(np.isfinite(scores).sum()),
        "mean_effective_k": float(kk),
        "overlap": overlap,
        "random_expected_overlap": chance,
        "chance_adjusted_overlap": overlap - chance if np.isfinite(overlap) else float("nan"),
        "chance_adjusted_unit_overlap": adjusted_unit(overlap, chance),
    }
    return row, scores


def pair_score_row(
    *,
    comparison: str,
    channel_a: str,
    channel_b: str,
    normalization: str,
    probe_neighbors: Mapping[str, np.ndarray],
    heldout_neighbors: Mapping[str, np.ndarray],
    k: int,
) -> tuple[dict[str, object], np.ndarray]:
    scores = per_query_cross_scores(
        probe_neighbors[channel_a],
        heldout_neighbors[channel_b],
        probe_neighbors[channel_b],
        heldout_neighbors[channel_a],
        k=int(k),
    )
    kk, chance = chance_level(probe_neighbors[channel_a], k=int(k))
    overlap = float(np.nanmean(scores)) if scores.size else float("nan")
    row = {
        "comparison": comparison,
        "channel_a": channel_a,
        "channel_b": channel_b,
        "normalization": normalization,
        "normalization_primary": bool(normalization_is_primary(normalization)),
        "normalization_transductive": bool(normalization_is_transductive(normalization)),
        "n_queries": int(probe_neighbors[channel_a].shape[0]),
        "n_valid_queries": int(np.isfinite(scores).sum()),
        "mean_effective_k": float(kk),
        "overlap": overlap,
        "random_expected_overlap": chance,
        "chance_adjusted_overlap": overlap - chance if np.isfinite(overlap) else float("nan"),
        "chance_adjusted_unit_overlap": adjusted_unit(overlap, chance),
    }
    return row, scores


def bootstrap_channel_row(
    base: Mapping[str, object],
    scores: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    mean, low, high = bootstrap_mean_ci(scores[np.isfinite(scores)], repeats=int(repeats), seed=int(seed))
    chance = float(base["random_expected_overlap"])
    return {
        **base,
        "overlap_mean": mean,
        "overlap_ci95_low": low,
        "overlap_ci95_high": high,
        "chance_adjusted_mean": mean - chance if np.isfinite(mean) else float("nan"),
        "chance_adjusted_ci95_low": low - chance if np.isfinite(low) else float("nan"),
        "chance_adjusted_ci95_high": high - chance if np.isfinite(high) else float("nan"),
        "chance_adjusted_unit_mean": adjusted_unit(mean, chance),
        "chance_adjusted_unit_ci95_low": adjusted_unit(low, chance),
        "chance_adjusted_unit_ci95_high": adjusted_unit(high, chance),
        "bootstrap_repeats": int(repeats),
    }


def bootstrap_paired_diff_ci(
    observed: np.ndarray,
    control: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float, float, int]:
    observed = np.asarray(observed, dtype=np.float32)
    control = np.asarray(control, dtype=np.float32)
    mask = np.isfinite(observed) & np.isfinite(control)
    diff = observed[mask] - control[mask]
    if diff.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(diff))
    if repeats <= 0 or diff.size == 1:
        return mean, float("nan"), float("nan"), int(diff.size)
    rng = np.random.default_rng(seed)
    samples = rng.choice(diff, size=(int(repeats), diff.size), replace=True)
    means = samples.mean(axis=1)
    return mean, float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), int(diff.size)


def feature_diagnostic_rows(
    features: Mapping[str, np.ndarray],
    *,
    feature_set: str,
    normalization: str,
    k: int,
    tie_tol: float = 1e-8,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for channel, x in features.items():
        x = np.asarray(x, dtype=np.float32)
        norms = np.linalg.norm(x, axis=1)
        zero_norm_fraction = float(np.mean(norms <= 1e-8)) if norms.size else float("nan")
        safe = x / np.maximum(norms[:, None], 1e-8)
        sim = safe @ safe.T
        if sim.shape[0] > 0:
            np.fill_diagonal(sim, -np.inf)
        finite = sim[np.isfinite(sim)]
        duplicate_fraction = float("nan")
        if x.shape[0] > 1:
            rounded = np.round(x, decimals=8)
            unique = np.unique(rounded, axis=0).shape[0]
            duplicate_fraction = 1.0 - float(unique) / float(x.shape[0])
        kth_margin_mean = float("nan")
        kth_margin_min = float("nan")
        boundary_tie_fraction = float("nan")
        if x.shape[0] > int(k) + 1 and finite.size:
            sorted_sim = np.sort(sim, axis=1)[:, ::-1]
            kth = sorted_sim[:, int(k) - 1]
            next_k = sorted_sim[:, int(k)]
            margin = kth - next_k
            kth_margin_mean = float(np.mean(margin))
            kth_margin_min = float(np.min(margin))
            boundary_tie_fraction = float(np.mean(margin <= float(tie_tol)))
        rows.append(
            {
                "feature_set": feature_set,
                "channel": channel,
                "normalization": normalization,
                "normalization_primary": bool(normalization_is_primary(normalization)),
                "normalization_transductive": bool(normalization_is_transductive(normalization)),
                "n_states": int(x.shape[0]),
                "feature_dim": int(x.shape[1]) if x.ndim == 2 else 0,
                "k": int(k),
                "zero_norm_fraction": zero_norm_fraction,
                "duplicate_fraction_rounded_1e8": duplicate_fraction,
                "kth_margin_mean": kth_margin_mean,
                "kth_margin_min": kth_margin_min,
                "boundary_tie_fraction": boundary_tie_fraction,
            }
        )
    return rows


def gate_summary_rows(
    identity_rows: Sequence[Mapping[str, object]],
    same_channel_rows: Sequence[Mapping[str, object]],
    cross_rows: Sequence[Mapping[str, object]],
    identity_boot_rows: Sequence[Mapping[str, object]],
    same_channel_boot_rows: Sequence[Mapping[str, object]],
    cross_boot_rows: Sequence[Mapping[str, object]],
    paired_diff_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_identity = {(str(r["normalization"]), str(r["channel"])): r for r in identity_rows}
    by_same = {(str(r["normalization"]), str(r["channel"])): r for r in same_channel_rows}
    by_cross = {
        (str(r["normalization"]), str(r["channel_a"]), str(r["channel_b"])): r
        for r in cross_rows
    }
    boot_identity = {(str(r["normalization"]), str(r["channel"])): r for r in identity_boot_rows}
    boot_same = {(str(r["normalization"]), str(r["channel"])): r for r in same_channel_boot_rows}
    boot_cross = {
        (str(r["normalization"]), str(r["channel_a"]), str(r["channel_b"])): r
        for r in cross_boot_rows
    }
    paired = {(str(r["normalization"]), str(r["channel"])): r for r in paired_diff_rows}
    rows: list[dict[str, object]] = []
    for norm in sorted({str(r["normalization"]) for r in identity_rows}):
        identity_channels = [row for key, row in by_identity.items() if key[0] == norm]
        same_channels_core = [
            by_same[(norm, ch)]
            for ch in CORE_RELIABILITY_CHANNELS
            if (norm, ch) in by_same
        ]
        same_channels_all = [
            by_same[(norm, ch)]
            for ch in ("rgb", "noisy_rgb", "gray_rgb", "blur_rgb")
            if (norm, ch) in by_same
        ]
        cross_pairs_core = [
            by_cross[(norm, a, b)]
            for a, b in CORE_REDUNDANCY_PAIRS
            if (norm, a, b) in by_cross
        ]
        cross_pairs_all = [
            by_cross[(norm, a, b)]
            for a, b in DEFAULT_REDUNDANCY_PAIRS
            if (norm, a, b) in by_cross
        ]
        identity_min = min((float(row["overlap"]) for row in identity_channels), default=float("nan"))
        same_core_min = min((float(row["chance_adjusted_overlap"]) for row in same_channels_core), default=float("nan"))
        same_all_min = min((float(row["chance_adjusted_overlap"]) for row in same_channels_all), default=float("nan"))
        cross_core_min = min((float(row["chance_adjusted_overlap"]) for row in cross_pairs_core), default=float("nan"))
        cross_all_min = min((float(row["chance_adjusted_overlap"]) for row in cross_pairs_all), default=float("nan"))
        same_core_ci_min = min(
            (
                float(boot_same[(norm, ch)]["chance_adjusted_ci95_low"])
                for ch in CORE_RELIABILITY_CHANNELS
                if (norm, ch) in boot_same
            ),
            default=float("nan"),
        )
        same_all_ci_min = min(
            (
                float(boot_same[(norm, ch)]["chance_adjusted_ci95_low"])
                for ch in ("rgb", "noisy_rgb", "gray_rgb", "blur_rgb")
                if (norm, ch) in boot_same
            ),
            default=float("nan"),
        )
        cross_core_ci_min = min(
            (
                float(boot_cross[(norm, a, b)]["chance_adjusted_ci95_low"])
                for a, b in CORE_REDUNDANCY_PAIRS
                if (norm, a, b) in boot_cross
            ),
            default=float("nan"),
        )
        cross_all_ci_min = min(
            (
                float(boot_cross[(norm, a, b)]["chance_adjusted_ci95_low"])
                for a, b in DEFAULT_REDUNDANCY_PAIRS
                if (norm, a, b) in boot_cross
            ),
            default=float("nan"),
        )
        paired_ci_min = min(
            (
                float(paired[(norm, ch)]["diff_ci95_low"])
                for ch in ("rgb", "noisy_rgb", "gray_rgb", "blur_rgb")
                if (norm, ch) in paired
            ),
            default=float("nan"),
        )
        identity_ci_min = min(
            (float(row["overlap_ci95_low"]) for key, row in boot_identity.items() if key[0] == norm),
            default=float("nan"),
        )
        rows.append(
            {
                "normalization": norm,
                "normalization_primary": bool(normalization_is_primary(norm)),
                "normalization_transductive": bool(normalization_is_transductive(norm)),
                "gate_minus1_identity_overlap_min": identity_min,
                "gate_minus1_identity_point_pass": bool(identity_min >= 0.999),
                "gate_minus1_identity_ci_low_min": identity_ci_min,
                "gate_minus1_identity_ci_pass": bool(identity_ci_min >= 0.999) if np.isfinite(identity_ci_min) else False,
                "gate_minus1_same_channel_core_adjusted_min": same_core_min,
                "gate_minus1_same_channel_core_point_pass": bool(same_core_min > 0.0),
                "gate_minus1_same_channel_core_ci_low_min": same_core_ci_min,
                "gate_minus1_same_channel_core_ci_pass": bool(same_core_ci_min > 0.0)
                if np.isfinite(same_core_ci_min)
                else False,
                "gate_minus1_same_channel_all_adjusted_min": same_all_min,
                "gate_minus1_same_channel_all_point_pass": bool(same_all_min > 0.0),
                "gate_minus1_same_channel_all_ci_low_min": same_all_ci_min,
                "gate_minus1_same_channel_all_ci_pass": bool(same_all_ci_min > 0.0)
                if np.isfinite(same_all_ci_min)
                else False,
                "gate_minus1_same_vs_shuffled_diff_ci_low_min": paired_ci_min,
                "gate_minus1_same_vs_shuffled_ci_pass": bool(paired_ci_min > 0.0) if np.isfinite(paired_ci_min) else False,
                "gate0_redundancy_core_adjusted_min": cross_core_min,
                "gate0_redundancy_core_point_pass": bool(cross_core_min > 0.0),
                "gate0_redundancy_core_ci_low_min": cross_core_ci_min,
                "gate0_redundancy_core_ci_pass": bool(cross_core_ci_min > 0.0)
                if np.isfinite(cross_core_ci_min)
                else False,
                "gate0_redundancy_all_adjusted_min": cross_all_min,
                "gate0_redundancy_all_point_pass": bool(cross_all_min > 0.0),
                "gate0_redundancy_all_ci_low_min": cross_all_ci_min,
                "gate0_redundancy_all_ci_pass": bool(cross_all_ci_min > 0.0)
                if np.isfinite(cross_all_ci_min)
                else False,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    data = load_npz(args.data)
    encoders, metadata = load_transition_encoders(args.model)
    if str(metadata.get("feature_prefix", "")) != "delta":
        raise ValueError("Stage B.4 reliability expects action-effect encoders with feature_prefix='delta'.")
    validate_model_data_path(metadata, args.data, allow_cross_data_model=bool(args.allow_cross_data_model))
    channels = [str(x) for x in args.channels]
    leakage = sorted(set(channels) & diagnostic_only_channels(data))
    if leakage:
        raise ValueError(f"Stage B.4 reliability gates exclude diagnostic-only channels: {leakage}")
    sample_indices = validate_dense_stage0_data(data, channels, require_detect=True)
    validate_model_fingerprint(metadata, data, channels, sample_indices, prefix="delta", allow_cross_data_model=bool(args.allow_cross_data_model))
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
    validate_probe_trained_model(
        metadata,
        [int(action_ids[i]) for i in probe_pos.tolist()],
        allow_all_action_trained_model=bool(args.allow_all_action_trained_model),
    )

    action_tensors = embedding_tensors(data, encoders, channels, prefix="delta", dense_rows=dense_rows)
    identity_rows: list[dict[str, object]] = []
    identity_boot_rows: list[dict[str, object]] = []
    same_channel_rows: list[dict[str, object]] = []
    same_channel_boot_rows: list[dict[str, object]] = []
    same_channel_shuffled_rows: list[dict[str, object]] = []
    same_channel_shuffled_boot_rows: list[dict[str, object]] = []
    same_vs_shuffled_rows: list[dict[str, object]] = []
    redundancy_rows: list[dict[str, object]] = []
    redundancy_boot_rows: list[dict[str, object]] = []
    feature_diag_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(int(args.bootstrap_seed))

    for norm in [str(x) for x in args.normalization_modes]:
        if norm not in NORMALIZATION_MODES:
            raise ValueError(f"Unsupported normalization mode: {norm}")
        ref_pos = probe_pos if norm in {"probe_global_apply", "probe_action_type_apply"} else None
        identity_a = features_from_tensors(
            action_tensors,
            channels,
            probe_pos,
            mode=norm,
            action_types=full_action_types,
            reference_positions=ref_pos,
        )
        identity_b = features_from_tensors(
            action_tensors,
            channels,
            probe_pos.copy(),
            mode=norm,
            action_types=full_action_types,
            reference_positions=ref_pos,
        )
        heldout_identity_a = features_from_tensors(
            action_tensors,
            channels,
            heldout_pos,
            mode=norm,
            action_types=full_action_types,
            reference_positions=ref_pos,
        )
        heldout_identity_b = features_from_tensors(
            action_tensors,
            channels,
            heldout_pos.copy(),
            mode=norm,
            action_types=full_action_types,
            reference_positions=ref_pos,
        )
        probe_features = identity_a
        heldout_features = features_from_tensors(
            action_tensors,
            channels,
            heldout_pos,
            mode=norm,
            action_types=full_action_types,
            reference_positions=ref_pos,
        )
        shuffled_heldout_features = shuffled_features_from_tensors(
            action_tensors,
            channels,
            heldout_pos,
            mode=norm,
            action_types=full_action_types,
            reference_positions=ref_pos,
            seed=int(args.seed) + 4001,
        )
        feature_diag_rows.extend(feature_diagnostic_rows(probe_features, feature_set="probe", normalization=norm, k=int(args.k)))
        feature_diag_rows.extend(
            feature_diagnostic_rows(heldout_features, feature_set="heldout", normalization=norm, k=int(args.k))
        )
        feature_diag_rows.extend(
            feature_diagnostic_rows(
                shuffled_heldout_features,
                feature_set="heldout_action_column_shuffled",
                normalization=norm,
                k=int(args.k),
            )
        )
        identity_a_neighbors = neighbor_sets(identity_a, k=int(args.k), batch_size=int(args.batch_size))
        identity_b_neighbors = neighbor_sets(identity_b, k=int(args.k), batch_size=int(args.batch_size))
        heldout_identity_a_neighbors = neighbor_sets(
            heldout_identity_a, k=int(args.k), batch_size=int(args.batch_size)
        )
        heldout_identity_b_neighbors = neighbor_sets(
            heldout_identity_b, k=int(args.k), batch_size=int(args.batch_size)
        )
        probe_neighbors = identity_a_neighbors
        heldout_neighbors = neighbor_sets(heldout_features, k=int(args.k), batch_size=int(args.batch_size))
        shuffled_heldout_neighbors = neighbor_sets(
            shuffled_heldout_features, k=int(args.k), batch_size=int(args.batch_size)
        )

        for channel in channels:
            for comparison, left, right in [
                ("identity_same_probe_action", identity_a_neighbors[channel], identity_b_neighbors[channel]),
                (
                    "identity_same_heldout_action",
                    heldout_identity_a_neighbors[channel],
                    heldout_identity_b_neighbors[channel],
                ),
            ]:
                base, scores = channel_score_row(
                    comparison=comparison,
                    channel=channel,
                    normalization=norm,
                    neighbors_a=left,
                    neighbors_b=right,
                    k=int(args.k),
                )
                add_common_context(
                    base,
                    k=int(args.k),
                    n_states=int(state_ids.shape[0]),
                    n_probe_actions=int(probe_pos.size),
                    n_heldout_actions=int(heldout_pos.size),
                )
                identity_rows.append(base)
                identity_boot_rows.append(
                    bootstrap_channel_row(
                        base,
                        scores,
                        repeats=int(args.bootstrap_repeats),
                        seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                    )
                )

            base, scores = channel_score_row(
                comparison="same_channel_probe_to_heldout",
                channel=channel,
                normalization=norm,
                neighbors_a=probe_neighbors[channel],
                neighbors_b=heldout_neighbors[channel],
                k=int(args.k),
            )
            add_common_context(
                base,
                k=int(args.k),
                n_states=int(state_ids.shape[0]),
                n_probe_actions=int(probe_pos.size),
                n_heldout_actions=int(heldout_pos.size),
            )
            same_channel_rows.append(base)
            same_channel_boot_rows.append(
                bootstrap_channel_row(
                    base,
                    scores,
                    repeats=int(args.bootstrap_repeats),
                    seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                )
            )
            shuffled_base, shuffled_scores = channel_score_row(
                comparison="same_channel_probe_to_shuffled_heldout",
                channel=channel,
                normalization=norm,
                neighbors_a=probe_neighbors[channel],
                neighbors_b=shuffled_heldout_neighbors[channel],
                k=int(args.k),
            )
            add_common_context(
                shuffled_base,
                k=int(args.k),
                n_states=int(state_ids.shape[0]),
                n_probe_actions=int(probe_pos.size),
                n_heldout_actions=int(heldout_pos.size),
            )
            same_channel_shuffled_rows.append(shuffled_base)
            same_channel_shuffled_boot_rows.append(
                bootstrap_channel_row(
                    shuffled_base,
                    shuffled_scores,
                    repeats=int(args.bootstrap_repeats),
                    seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                )
            )
            diff_mean, diff_low, diff_high, n_diff = bootstrap_paired_diff_ci(
                scores,
                shuffled_scores,
                repeats=int(args.bootstrap_repeats),
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
            same_vs_shuffled_rows.append(
                {
                    "comparison": "same_channel_minus_action_column_shuffled",
                    "channel": channel,
                    "normalization": norm,
                    "normalization_primary": bool(normalization_is_primary(norm)),
                    "normalization_transductive": bool(normalization_is_transductive(norm)),
                    "k": int(args.k),
                    "n_states": int(state_ids.shape[0]),
                    "n_probe_actions": int(probe_pos.size),
                    "n_heldout_actions": int(heldout_pos.size),
                    "n_valid_queries": int(n_diff),
                    "diff_mean": diff_mean,
                    "diff_ci95_low": diff_low,
                    "diff_ci95_high": diff_high,
                    "bootstrap_repeats": int(args.bootstrap_repeats),
                }
            )

        for channel_a, channel_b in DEFAULT_REDUNDANCY_PAIRS:
            if channel_a not in channels or channel_b not in channels:
                continue
            base, scores = pair_score_row(
                comparison="redundancy_probe_to_heldout_cross",
                channel_a=channel_a,
                channel_b=channel_b,
                normalization=norm,
                probe_neighbors=probe_neighbors,
                heldout_neighbors=heldout_neighbors,
                k=int(args.k),
            )
            add_common_context(
                base,
                k=int(args.k),
                n_states=int(state_ids.shape[0]),
                n_probe_actions=int(probe_pos.size),
                n_heldout_actions=int(heldout_pos.size),
            )
            redundancy_rows.append(base)
            redundancy_boot_rows.append(
                bootstrap_channel_row(
                    base,
                    scores,
                    repeats=int(args.bootstrap_repeats),
                    seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                )
            )

    out = ensure_dir(args.out)
    reports = ensure_dir(out / "reports")
    write_csv(reports / "identity_same_action_reliability.csv", identity_rows)
    write_csv(reports / "identity_same_action_bootstrap_ci.csv", identity_boot_rows)
    write_csv(reports / "same_channel_probe_heldout_reliability.csv", same_channel_rows)
    write_csv(reports / "same_channel_probe_heldout_bootstrap_ci.csv", same_channel_boot_rows)
    write_csv(reports / "same_channel_shuffled_reliability.csv", same_channel_shuffled_rows)
    write_csv(reports / "same_channel_shuffled_bootstrap_ci.csv", same_channel_shuffled_boot_rows)
    write_csv(reports / "same_channel_vs_shuffled_paired_ci.csv", same_vs_shuffled_rows)
    write_csv(reports / "redundancy_probe_heldout_calibration.csv", redundancy_rows)
    write_csv(reports / "redundancy_probe_heldout_bootstrap_ci.csv", redundancy_boot_rows)
    write_csv(reports / "feature_tie_diagnostics.csv", feature_diag_rows)
    gate_rows = gate_summary_rows(
        identity_rows,
        same_channel_rows,
        redundancy_rows,
        identity_boot_rows,
        same_channel_boot_rows,
        redundancy_boot_rows,
        same_vs_shuffled_rows,
    )
    write_csv(reports / "gate_summary_ci.csv", gate_rows)
    save_json(
        {
            "data": str(Path(args.data)),
            "model": str(Path(args.model)),
            "channels": channels,
            "normalization_modes": [str(x) for x in args.normalization_modes],
            "primary_normalization_modes": sorted(PRIMARY_NORMALIZATION_MODES),
            "k": int(args.k),
            "n_states": int(state_ids.shape[0]),
            "n_actions": int(action_ids.shape[0]),
            "probe_action_ids": [int(action_ids[i]) for i in probe_pos.tolist()],
            "heldout_action_ids": [int(action_ids[i]) for i in heldout_pos.tolist()],
            "bootstrap_repeats": int(args.bootstrap_repeats),
                "gate_order": [
                "gate_minus1_identity_same_probe_and_heldout_action",
                "gate_minus1_same_channel_probe_heldout",
                "gate0_redundancy_cross_channel_calibration",
                "gate1_rgb_range_action_effect_gt_static",
                "gate2_binary_vs_continuous_observability",
            ],
            "decision_table": {
                "identity_fails": "pipeline bug before any cross-action interpretation",
                "same_channel_fails": "repair action bank / signature / encoder / normalization before cross-channel alignment",
                "redundancy_fails": "repair cross-channel calibration before rgb-range interpretation",
                "redundancy_passes": "run Gate 1/2 with Stage B.3 metrics under the passing normalization",
            },
        },
        reports / "stageb4_summary.json",
    )
    print(f"Wrote Stage B.4 reliability reports to {out}")


if __name__ == "__main__":
    main()
