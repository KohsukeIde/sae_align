#!/usr/bin/env python
"""Analyze Stage B action-effect kNN alignment by strata."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import argparse
import csv
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

from sae_align.analysis.knn_alignment import (
    action_confound_features,
    cosine_knn_indices,
    pairwise_overlap_matrix,
    pairwise_stratified_overlap_rows,
    residualize_embeddings,
    restricted_candidate_counts,
    restricted_cosine_knn_indices,
    shuffled_assignment_knn_indices,
    stratified_knn_summary,
)
from sae_align.analysis.strata import (
    channel_blind_masks,
    dense_delta_sample_indices,
    diagnostic_only_channels,
    effect_bins,
    named_strata,
    pair_channel_strata,
    subset_stage0_vector,
    validate_dense_stage0_data,
)
from sae_align.models import load_transition_encoders
from sae_align.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--channels", nargs="*", default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-points", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--physical-quantile", type=float, default=0.10)
    p.add_argument("--detectability-quantile", type=float, default=0.10)
    p.add_argument("--effect-quantiles", nargs="*", type=float, default=[0.33, 0.66])
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument(
        "--allow-leakage-diagnostic",
        action="store_true",
        help="Allow diagnostic-only post-action channels such as event_response in this diagnostic-only analysis.",
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


def plot_matrix(path: Path, names: Sequence[str], mat: np.ndarray, title: str) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(names)), max(4, 0.65 * len(names))))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def action_only_features(data: Dict[str, np.ndarray], sample_indices: np.ndarray) -> np.ndarray | None:
    if "action_array" not in data:
        return None
    action_array = subset_stage0_vector(data, "action_array", sample_indices)
    action_type = subset_stage0_vector(data, "action_type", sample_indices).astype(str)
    return action_confound_features(action_array, action_type)


def action_id_labels(data: Dict[str, np.ndarray], sample_indices: np.ndarray) -> tuple[np.ndarray, str] | None:
    if "action_id" in data:
        return subset_stage0_vector(data, "action_id", sample_indices).astype(np.int64), "same_action_id"
    if "action_array" not in data:
        return None
    action_array = np.asarray(subset_stage0_vector(data, "action_array", sample_indices), dtype=np.float32)
    if action_array.ndim == 1:
        action_array = action_array[:, None]
    else:
        action_array = action_array.reshape(action_array.shape[0], -1)
    action_type = subset_stage0_vector(data, "action_type", sample_indices).astype(str)
    labels: list[int] = []
    seen: dict[tuple[object, ...], int] = {}
    for typ, row in zip(action_type.tolist(), action_array.tolist()):
        key = (typ, *row)
        if key not in seen:
            seen[key] = len(seen)
        labels.append(seen[key])
    return np.asarray(labels, dtype=np.int64), "same_action_signature"


def action_control_rows(
    neighbor_sets: Dict[str, np.ndarray],
    action_neighbors: np.ndarray,
    pair_strata: Dict[tuple[str, str], Dict[str, np.ndarray]],
    *,
    k: int,
    control: str = "action_only",
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    n = action_neighbors.shape[0]
    chance = float(min(k, action_neighbors.shape[1]) / max(1, n - 1)) if n > 1 else float("nan")
    for ch, neighbors in neighbor_sets.items():
        channel_strata = {"all": np.ones(n, dtype=bool)}
        for (a, b), strata in pair_strata.items():
            if ch == a or ch == b:
                for name, mask in strata.items():
                    channel_strata.setdefault(name, np.asarray(mask, dtype=bool))
        for stratum, mask in channel_strata.items():
            from sae_align.analysis.knn_alignment import neighbor_overlap

            overlap = neighbor_overlap(neighbors, action_neighbors, k=k, query_mask=mask)
            rows.append(
                {
                    "control": control,
                    "channel": ch,
                    "stratum": stratum,
                    "n_queries": int(np.asarray(mask, dtype=bool).sum()),
                    "overlap": overlap,
                    "random_expected_overlap": chance,
                    "chance_adjusted_overlap": float(overlap - chance) if np.isfinite(overlap) else float("nan"),
                }
            )
    return rows


def shuffled_pair_strata(
    pair_strata: Dict[tuple[str, str], Dict[str, np.ndarray]],
    seed: int,
) -> Dict[tuple[str, str], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    shuffled: Dict[tuple[str, str], Dict[str, np.ndarray]] = {}
    for pair, strata in pair_strata.items():
        shuffled[pair] = {}
        for name, mask in strata.items():
            arr = np.asarray(mask, dtype=bool)
            if name == "all":
                shuffled[pair][name] = arr.copy()
            else:
                shuffled[pair][name] = arr[rng.permutation(arr.shape[0])]
    return shuffled


def main() -> None:
    args = parse_args()
    data = load_npz(args.data)
    encoders, metadata = load_transition_encoders(args.model)
    channels = [str(x) for x in (args.channels or list(encoders.keys()))]
    missing_model = [ch for ch in channels if ch not in encoders]
    missing_data = [ch for ch in channels if f"delta_{ch}" not in data]
    if missing_model or missing_data:
        raise KeyError(f"Missing model channels={missing_model}, data deltas={missing_data}")
    leakage_channels = sorted(set(channels) & diagnostic_only_channels(data))
    if leakage_channels and not args.allow_leakage_diagnostic:
        raise ValueError(
            "Diagnostic-only post-action channels are excluded from primary Stage B analysis: "
            f"{leakage_channels}. Re-run with --allow-leakage-diagnostic only for separately labeled diagnostics."
        )

    sample_indices = validate_dense_stage0_data(data, channels, require_detect=True)
    n_dense = int(sample_indices.shape[0])
    dense_keep = np.arange(n_dense)
    if n_dense > int(args.max_points):
        rng = np.random.default_rng(args.seed)
        dense_keep = np.sort(rng.choice(n_dense, size=int(args.max_points), replace=False))
        sample_indices = sample_indices[dense_keep]

    world_delta = subset_stage0_vector(data, "world_delta", sample_indices).astype(np.float32)
    action_type = subset_stage0_vector(data, "action_type", sample_indices).astype(str)
    effect_bin, effect_cuts = effect_bins(world_delta, quantiles=args.effect_quantiles)
    strata = named_strata(
        world_delta,
        action_type,
        physical_q=float(args.physical_quantile),
        effect_quantiles=args.effect_quantiles,
    )

    neighbor_sets: Dict[str, np.ndarray] = {}
    embedding_sets: Dict[str, np.ndarray] = {}
    shuffled_neighbor_sets: Dict[str, np.ndarray] = {}
    rng = np.random.default_rng(args.seed + 2027)
    for ch in channels:
        emb = encoders[ch].transform(data[f"delta_{ch}"][dense_keep])
        embedding_sets[ch] = emb
        neighbor_sets[ch] = cosine_knn_indices(emb, k=int(args.k), batch_size=int(args.batch_size))
        perm = rng.permutation(emb.shape[0])
        shuffled_neighbor_sets[ch] = shuffled_assignment_knn_indices(
            emb,
            perm,
            k=int(args.k),
            batch_size=int(args.batch_size),
        )

    overlap_names, overlap = pairwise_overlap_matrix(neighbor_sets, k=int(args.k))
    rows = stratified_knn_summary(
        neighbor_sets,
        action_type=action_type,
        effect_bin=effect_bin,
        world_delta=world_delta,
        strata=strata,
    )
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
    pair_rows = pairwise_stratified_overlap_rows(neighbor_sets, pair_strata, k=int(args.k), control="observed")
    shuffled_pair_rows = pairwise_stratified_overlap_rows(
        shuffled_neighbor_sets,
        pair_strata,
        k=int(args.k),
        control="shuffled_embeddings",
    )
    shuffled_strata_rows = pairwise_stratified_overlap_rows(
        neighbor_sets,
        shuffled_pair_strata(pair_strata, seed=args.seed + 404),
        k=int(args.k),
        control="shuffled_strata",
    )
    all_pair_rows = pair_rows + shuffled_pair_rows + shuffled_strata_rows

    same_action_type_neighbors = {
        ch: restricted_cosine_knn_indices(emb, action_type, k=int(args.k), batch_size=int(args.batch_size))
        for ch, emb in embedding_sets.items()
    }
    same_action_type_counts = restricted_candidate_counts(action_type)
    same_action_type_rows = pairwise_stratified_overlap_rows(
        same_action_type_neighbors,
        pair_strata,
        k=int(args.k),
        control="same_action_type",
        candidate_counts=same_action_type_counts,
    )
    all_pair_rows.extend(same_action_type_rows)

    same_action_id_rows: list[Dict[str, object]] = []
    same_action_signature_rows: list[Dict[str, object]] = []
    action_id_info = action_id_labels(data, sample_indices)
    if action_id_info is not None:
        action_ids, action_id_control = action_id_info
        same_action_id_neighbors = {
            ch: restricted_cosine_knn_indices(emb, action_ids, k=int(args.k), batch_size=int(args.batch_size))
            for ch, emb in embedding_sets.items()
        }
        rows_for_action_id_control = pairwise_stratified_overlap_rows(
            same_action_id_neighbors,
            pair_strata,
            k=int(args.k),
            control=action_id_control,
            candidate_counts=restricted_candidate_counts(action_ids),
        )
        if action_id_control == "same_action_id":
            same_action_id_rows = rows_for_action_id_control
        else:
            same_action_signature_rows = rows_for_action_id_control
        all_pair_rows.extend(rows_for_action_id_control)

    action_rows: list[Dict[str, object]] = []
    action_x = action_only_features(data, sample_indices)
    if action_x is not None:
        action_neighbors = cosine_knn_indices(action_x, k=int(args.k), batch_size=int(args.batch_size))
        action_rows = action_control_rows(neighbor_sets, action_neighbors, pair_strata, k=int(args.k))
        shuffled_action_x = action_x[np.random.default_rng(args.seed + 505).permutation(action_x.shape[0])]
        shuffled_action_neighbors = cosine_knn_indices(shuffled_action_x, k=int(args.k), batch_size=int(args.batch_size))
        action_rows.extend(
            action_control_rows(
                neighbor_sets,
                shuffled_action_neighbors,
                pair_strata,
                k=int(args.k),
                control="shuffled_action",
            )
        )

    residualized_rows: list[Dict[str, object]] = []
    residualization_diagnostics: list[Dict[str, object]] = []
    residualized_overlap_names: Sequence[str] = []
    residualized_overlap = np.zeros((0, 0), dtype=np.float32)
    if action_x is not None:
        residualized_neighbor_sets = {}
        for ch, emb in embedding_sets.items():
            resid = residualize_embeddings(emb, action_x)
            residualized_neighbor_sets[ch] = cosine_knn_indices(
                resid,
                k=int(args.k),
                batch_size=int(args.batch_size),
            )
            original_energy = float(np.mean(np.sum(np.asarray(emb, dtype=np.float32) ** 2, axis=1)))
            residual_energy = float(np.mean(np.sum(np.asarray(resid, dtype=np.float32) ** 2, axis=1)))
            residualization_diagnostics.append(
                {
                    "channel": ch,
                    "original_mean_squared_norm": original_energy,
                    "residual_mean_squared_norm": residual_energy,
                    "removed_fraction": 1.0 - residual_energy / original_energy
                    if original_energy > 0.0
                    else float("nan"),
                }
            )
        residualized_overlap_names, residualized_overlap = pairwise_overlap_matrix(
            residualized_neighbor_sets,
            k=int(args.k),
        )
        residualized_rows = pairwise_stratified_overlap_rows(
            residualized_neighbor_sets,
            pair_strata,
            k=int(args.k),
            control="action_residualized",
        )
        all_pair_rows.extend(residualized_rows)

    out = ensure_dir(args.out)
    report_dir = ensure_dir(out / "reports")
    fig_dir = ensure_dir(out / "figures")
    write_matrix_csv(report_dir / "knn_overlap.csv", overlap_names, overlap)
    if len(residualized_overlap_names):
        write_matrix_csv(report_dir / "action_residualized_knn_overlap.csv", residualized_overlap_names, residualized_overlap)
    write_long_csv(report_dir / "neighbor_label_purity.csv", rows)
    write_long_csv(report_dir / "stratified_knn.csv", rows)
    write_long_csv(report_dir / "alignment_by_pair_and_stratum.csv", all_pair_rows)
    write_long_csv(report_dir / "pairwise_stratified_overlap.csv", all_pair_rows)
    write_long_csv(report_dir / "action_only_overlap.csv", action_rows)
    write_long_csv(report_dir / "same_action_type_restricted_knn.csv", same_action_type_rows)
    write_long_csv(report_dir / "same_action_id_restricted_knn.csv", same_action_id_rows)
    write_long_csv(report_dir / "same_action_signature_restricted_knn.csv", same_action_signature_rows)
    write_long_csv(report_dir / "action_residualized_pairwise_stratified_overlap.csv", residualized_rows)
    write_long_csv(report_dir / "action_residualization_diagnostics.csv", residualization_diagnostics)
    plot_matrix(fig_dir / "knn_overlap.png", overlap_names, overlap, "Stage B kNN overlap")

    summary = {
        "data": str(Path(args.data)),
        "model": str(Path(args.model)),
        "model_metadata": metadata,
        "channels": channels,
        "k": int(args.k),
        "n_points": int(sample_indices.shape[0]),
        "sample_indices_min": int(sample_indices.min()) if sample_indices.size else None,
        "sample_indices_max": int(sample_indices.max()) if sample_indices.size else None,
        "effect_quantiles": [float(x) for x in args.effect_quantiles],
        "effect_cuts": [float(x) for x in effect_cuts.tolist()],
        "detectability_quantile": float(args.detectability_quantile),
        "strata_thresholds": thresholds,
        "strata_counts": {
            f"{a}__{b}__{name}": int(mask.sum())
            for (a, b), strata_dict in pair_strata.items()
            for name, mask in strata_dict.items()
        },
        "controls": {
            "shuffled_embeddings": bool(shuffled_pair_rows),
            "shuffled_strata": bool(shuffled_strata_rows),
            "action_only": bool(action_rows),
            "shuffled_action": bool(action_rows),
            "same_action_type": bool(same_action_type_rows),
            "same_action_id": bool(same_action_id_rows),
            "same_action_signature": bool(same_action_signature_rows),
            "action_residualized": bool(residualized_rows),
        },
        "mean_pairwise_overlap": float(np.nanmean(overlap[np.triu_indices(len(channels), k=1)]))
        if len(channels) > 1
        else float("nan"),
    }
    save_json(summary, report_dir / "stageb_knn_summary.json")
    print(f"Wrote Stage B kNN reports to {out}")


if __name__ == "__main__":
    main()
