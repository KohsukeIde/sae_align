#!/usr/bin/env python
"""Compare static and action-effect kNN alignment on the same Stage 0 strata."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import argparse
import csv
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from sae_align.analysis.knn_alignment import (
    cosine_knn_indices,
    excluded_candidate_counts,
    excluded_cosine_knn_indices,
    pairwise_overlap_matrix,
    pairwise_stratified_overlap_rows,
)
from sae_align.analysis.strata import (
    channel_blind_masks,
    diagnostic_only_channels,
    pair_channel_strata,
    validate_dense_stage0_data,
    validate_dense_static_data,
)
from sae_align.models import load_transition_encoders
from sae_align.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--action-effect-model", type=str, required=True)
    p.add_argument("--static-model", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--channels", nargs="*", default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-points", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--detectability-quantile", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument(
        "--allow-same-state-static-neighbors",
        action="store_true",
        help="Allow static kNN to retrieve repeated state-action rows from the same state. Off by default to avoid duplicate-state leakage.",
    )
    p.add_argument(
        "--allow-leakage-diagnostic",
        action="store_true",
        help="Allow diagnostic-only post-action channels. Do not use such rows as primary Stage B evidence.",
    )
    return p.parse_args()


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def write_long_csv(path: Path, rows: list[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path: Path, names: Sequence[str], mat: np.ndarray) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + list(names))
        for name, row in zip(names, mat):
            writer.writerow([name] + [f"{float(x):.6f}" if np.isfinite(x) else "nan" for x in row])


def aligned_dense_rows(
    action_indices: np.ndarray,
    static_indices: np.ndarray,
    *,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action_pos = {int(sample_id): i for i, sample_id in enumerate(np.asarray(action_indices, dtype=np.int64).tolist())}
    static_pos = {int(sample_id): i for i, sample_id in enumerate(np.asarray(static_indices, dtype=np.int64).tolist())}
    common = np.asarray(sorted(set(action_pos) & set(static_pos)), dtype=np.int64)
    if common.size == 0:
        raise ValueError("Action-effect and static dense subsets have no shared sample IDs.")
    if common.size > int(max_points):
        rng = np.random.default_rng(seed)
        common = np.sort(rng.choice(common, size=int(max_points), replace=False))
    action_rows = np.asarray([action_pos[int(x)] for x in common], dtype=np.int64)
    static_rows = np.asarray([static_pos[int(x)] for x in common], dtype=np.int64)
    return common, action_rows, static_rows


def choose_channels(
    requested: Sequence[str] | None,
    action_encoders: Mapping[str, object],
    static_encoders: Mapping[str, object],
    data: Mapping[str, np.ndarray],
) -> list[str]:
    if requested:
        channels = [str(x) for x in requested]
    else:
        static_set = set(static_encoders)
        channels = [ch for ch in action_encoders if ch in static_set]
    missing = []
    for ch in channels:
        if ch not in action_encoders:
            missing.append(f"action-effect model:{ch}")
        if ch not in static_encoders:
            missing.append(f"static model:{ch}")
        if f"delta_{ch}" not in data:
            missing.append(f"data:delta_{ch}")
        if f"obs0_{ch}" not in data:
            missing.append(f"data:obs0_{ch}")
        if f"detect_{ch}" not in data:
            missing.append(f"data:detect_{ch}")
    if missing:
        raise KeyError(f"Missing channels or arrays: {missing}")
    if not channels:
        raise ValueError("No shared channels selected.")
    return channels


def feature_rows(
    data: Mapping[str, np.ndarray],
    encoders: Mapping[str, object],
    channels: Sequence[str],
    *,
    prefix: str,
    dense_rows: np.ndarray,
    pair_strata: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    k: int,
    batch_size: int,
    control: str,
    exclude_labels: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray, list[Dict[str, object]]]:
    neighbor_sets = {}
    for ch in channels:
        emb = encoders[ch].transform(data[f"{prefix}_{ch}"][dense_rows])
        if exclude_labels is None:
            neighbor_sets[ch] = cosine_knn_indices(emb, k=k, batch_size=batch_size)
        else:
            neighbor_sets[ch] = excluded_cosine_knn_indices(emb, exclude_labels, k=k, batch_size=batch_size)
    names, overlap = pairwise_overlap_matrix(neighbor_sets, k=k)
    candidate_counts = None if exclude_labels is None else excluded_candidate_counts(exclude_labels)
    rows = pairwise_stratified_overlap_rows(
        neighbor_sets,
        pair_strata,
        k=k,
        control=control,
        candidate_counts=candidate_counts,
    )
    for row in rows:
        row["feature_kind"] = control
    return names, overlap, rows


def gain_rows(action_rows: list[Dict[str, object]], static_rows: list[Dict[str, object]]) -> list[Dict[str, object]]:
    def key(row: Mapping[str, object]) -> tuple[object, object, object]:
        return (row["channel_a"], row["channel_b"], row["stratum"])

    action_by_key = {key(row): row for row in action_rows}
    static_by_key = {key(row): row for row in static_rows}
    rows: list[Dict[str, object]] = []
    for item in sorted(set(action_by_key) & set(static_by_key)):
        arow = action_by_key[item]
        srow = static_by_key[item]
        action_overlap = float(arow["overlap"])
        static_overlap = float(srow["overlap"])
        action_adj = float(arow["chance_adjusted_overlap"])
        static_adj = float(srow["chance_adjusted_overlap"])
        rows.append(
            {
                "channel_a": item[0],
                "channel_b": item[1],
                "stratum": item[2],
                "n_queries": int(arow["n_queries"]),
                "static_overlap": static_overlap,
                "action_effect_overlap": action_overlap,
                "delta_minus_static_gain": action_overlap - static_overlap
                if np.isfinite(action_overlap) and np.isfinite(static_overlap)
                else float("nan"),
                "static_chance_adjusted_overlap": static_adj,
                "action_effect_chance_adjusted_overlap": action_adj,
                "chance_adjusted_gain": action_adj - static_adj
                if np.isfinite(action_adj) and np.isfinite(static_adj)
                else float("nan"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    data = load_npz(args.data)
    action_encoders, action_meta = load_transition_encoders(args.action_effect_model)
    static_encoders, static_meta = load_transition_encoders(args.static_model)

    if action_meta.get("feature_prefix", "delta") != "delta":
        raise ValueError("Expected --action-effect-model metadata feature_prefix='delta'.")
    if static_meta.get("feature_prefix") != "obs0":
        raise ValueError("Expected --static-model metadata feature_prefix='obs0'.")

    channels = choose_channels(args.channels, action_encoders, static_encoders, data)
    leakage_channels = sorted(set(channels) & diagnostic_only_channels(data))
    if leakage_channels and not args.allow_leakage_diagnostic:
        raise ValueError(
            "Diagnostic-only channels are excluded from primary static/action-effect comparison: "
            f"{leakage_channels}."
        )

    action_indices = validate_dense_stage0_data(data, channels, require_detect=True)
    static_indices = validate_dense_static_data(data, channels)
    sample_indices, action_rows_idx, static_rows_idx = aligned_dense_rows(
        action_indices,
        static_indices,
        max_points=int(args.max_points),
        seed=int(args.seed),
    )
    state_labels = None
    if not args.allow_same_state_static_neighbors and "state_id" in data:
        state_labels = np.asarray(data["state_id"], dtype=np.int64)[sample_indices]

    physical, blind, regular, thresholds = channel_blind_masks(
        data,
        channels,
        sample_indices,
        threshold_quantile=float(args.detectability_quantile),
    )
    pair_strata = {
        (a, b): pair_channel_strata(physical, blind, regular, a, b)
        for i, a in enumerate(channels)
        for b in channels[i + 1 :]
    }

    action_names, action_overlap, action_pair_rows = feature_rows(
        data,
        action_encoders,
        channels,
        prefix="delta",
        dense_rows=action_rows_idx,
        pair_strata=pair_strata,
        k=int(args.k),
        batch_size=int(args.batch_size),
        control="action_effect",
    )
    static_names, static_overlap, static_pair_rows = feature_rows(
        data,
        static_encoders,
        channels,
        prefix="obs0",
        dense_rows=static_rows_idx,
        pair_strata=pair_strata,
        k=int(args.k),
        batch_size=int(args.batch_size),
        control="static",
        exclude_labels=state_labels,
    )
    gain = gain_rows(action_pair_rows, static_pair_rows)

    out = ensure_dir(args.out)
    report_dir = ensure_dir(out / "reports")
    write_matrix_csv(report_dir / "action_effect_knn_overlap.csv", action_names, action_overlap)
    write_matrix_csv(report_dir / "static_knn_overlap.csv", static_names, static_overlap)
    write_long_csv(report_dir / "action_effect_knn.csv", action_pair_rows)
    write_long_csv(report_dir / "static_knn.csv", static_pair_rows)
    write_long_csv(report_dir / "static_vs_action_effect_knn.csv", static_pair_rows + action_pair_rows)
    write_long_csv(report_dir / "delta_minus_static_gain.csv", gain)
    save_json(
        {
            "data": str(Path(args.data)),
            "action_effect_model": str(Path(args.action_effect_model)),
            "static_model": str(Path(args.static_model)),
            "action_effect_metadata": action_meta,
            "static_metadata": static_meta,
            "channels": channels,
            "k": int(args.k),
            "n_points": int(sample_indices.shape[0]),
            "sample_indices_min": int(sample_indices.min()) if sample_indices.size else None,
            "sample_indices_max": int(sample_indices.max()) if sample_indices.size else None,
            "detectability_quantile": float(args.detectability_quantile),
            "same_state_static_neighbors_allowed": bool(args.allow_same_state_static_neighbors),
            "same_state_static_neighbors_excluded": state_labels is not None,
            "strata_thresholds": thresholds,
        },
        report_dir / "static_action_effect_summary.json",
    )
    print(f"Wrote static/action-effect kNN comparison to {out}")


if __name__ == "__main__":
    main()
