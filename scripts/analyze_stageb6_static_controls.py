#!/usr/bin/env python
"""Posthoc static-control diagnostics for Stage B.6.

This script is intentionally narrower than the main B.6 analyzer.  It asks
whether the real-Powderworld action-effect signal survives after controlling
for static state similarity.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_SRC = _SCRIPT_DIR.parents[0] / "src"
for _p in (str(_SCRIPT_DIR), str(_REPO_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_stageb3_gates import NORMALIZATION_MODES, features_from_tensors  # noqa: E402
from analyze_stageb6_diagnostics import (  # noqa: E402
    calibrated_cknna,
    calibrated_cycle_knn,
    normalized_then_shuffled_features_from_tensors,
    validate_action_type_columns,
)
from analyze_stageb5_heldout_alignment import embedding_tensors_from_model  # noqa: E402
from analyze_state_signature_knn import complete_state_action_matrix, dense_rows_for_sample_ids  # noqa: E402
from sae_align.analysis.knn_alignment import cosine_knn_indices  # noqa: E402
from sae_align.analysis.strata import (  # noqa: E402
    diagnostic_only_channels,
    validate_dense_stage0_data,
    validate_dense_static_data,
)
from sae_align.models import load_transition_encoders  # noqa: E402
from sae_align.utils.io import ensure_dir  # noqa: E402


DEFAULT_PAIRS = ("rgb:range",)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True, help="Stage B.6 root with seed_*/split_*/pca_* runs.")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--channels", nargs="*", default=None)
    p.add_argument("--target-pairs", nargs="*", default=list(DEFAULT_PAIRS))
    p.add_argument("--pca-dims", nargs="*", type=int, default=None)
    p.add_argument("--data-seeds", nargs="*", type=int, default=None)
    p.add_argument("--split-seeds", nargs="*", type=int, default=None)
    p.add_argument("--normalization-modes", nargs="*", default=["probe_action_type_apply"])
    p.add_argument("--k-values", nargs="*", type=int, default=[5, 10, 20])
    p.add_argument("--max-states", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--residual-ridge-alpha", type=float, default=1.0)
    p.add_argument("--crossfit-folds", type=int, default=5)
    p.add_argument("--static-bins", type=int, default=4)
    p.add_argument("--literature-k", type=int, default=10)
    p.add_argument("--permutation-repeats", type=int, default=50)
    p.add_argument("--skip-literature-metrics", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--expected-report-dirs", type=int, default=None)
    return p.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_pairs(values: Sequence[str]) -> list[tuple[str, str]]:
    out = []
    for value in values:
        if ":" not in str(value):
            raise ValueError(f"Expected pair channel_a:channel_b, got {value!r}")
        a, b = str(value).split(":", 1)
        out.append((a, b))
    return out


def read_probe_ids(path: Path) -> list[int]:
    return [int(x) for x in path.read_text(encoding="utf-8").split()]


def load_static_control_npz(path: Path, channels: Sequence[str]) -> dict[str, np.ndarray]:
    needed = {
        "channels",
        "state_id",
        "action_id",
        "action_type",
        "world_delta",
        "delta_sample_indices",
        "static_obs_sample_indices",
    }
    for channel in channels:
        needed.add(f"delta_{channel}")
        needed.add(f"obs0_{channel}")
        needed.add(f"detect_{channel}")
    out: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=True) as f:
        available = set(f.files)
        for key in needed:
            if key in available:
                out[key] = f[key]
    return out


def run_metadata(run: Path, root: Path) -> dict[str, object]:
    rel = run.relative_to(root)
    meta: dict[str, object] = {"run": str(run)}
    for part in rel.parts:
        if part.startswith("seed_"):
            meta["data_seed"] = part.removeprefix("seed_")
        elif part.startswith("split_"):
            meta["split_seed"] = part.removeprefix("split_")
        elif part.startswith("pca_"):
            meta["pca_dim"] = part.removeprefix("pca_")
    return meta


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def per_query_overlap(neigh_a: np.ndarray, neigh_b: np.ndarray, k: int) -> np.ndarray:
    kk = min(int(k), neigh_a.shape[1], neigh_b.shape[1])
    scores = []
    for i in range(neigh_a.shape[0]):
        a = [int(v) for v in neigh_a[i, :kk].tolist() if int(v) >= 0]
        b = [int(v) for v in neigh_b[i, :kk].tolist() if int(v) >= 0]
        denom = min(kk, len(a), len(b))
        scores.append(len(set(a) & set(b)) / float(denom) if denom else float("nan"))
    return np.asarray(scores, dtype=np.float32)


def overlap_summary(features: Mapping[str, np.ndarray], pair: tuple[str, str], k: int, batch_size: int) -> tuple[dict[str, object], np.ndarray]:
    a, b = pair
    neigh_a = cosine_knn_indices(features[a], k=int(k), batch_size=int(batch_size))
    neigh_b = cosine_knn_indices(features[b], k=int(k), batch_size=int(batch_size))
    scores = per_query_overlap(neigh_a, neigh_b, int(k))
    n = int(neigh_a.shape[0])
    kk = min(int(k), int(neigh_a.shape[1]), int(neigh_b.shape[1]))
    chance = float(kk / max(1, n - 1)) if n > 1 else float("nan")
    overlap = float(np.nanmean(scores)) if scores.size else float("nan")
    return (
        {
            "n_queries": n,
            "k": int(k),
            "mean_effective_k": kk,
            "overlap": overlap,
            "random_expected_overlap": chance,
            "chance_adjusted_overlap": overlap - chance if np.isfinite(overlap) else float("nan"),
            "n_valid_queries": int(np.isfinite(scores).sum()),
        },
        scores,
    )


def ridge_residualize(y: np.ndarray, x: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    design = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    reg = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + reg, design.T @ y)
    pred = design @ beta
    resid = y - pred
    frac = float(np.linalg.norm(resid) / (np.linalg.norm(y - y.mean(axis=0, keepdims=True)) + 1e-12))
    return resid.astype(np.float32), frac


def ridge_fit_apply(
    y_train: np.ndarray,
    x_train: np.ndarray,
    y_eval: np.ndarray,
    x_eval: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    y_train = np.asarray(y_train, dtype=np.float64)
    x_train = np.asarray(x_train, dtype=np.float64)
    y_eval = np.asarray(y_eval, dtype=np.float64)
    x_eval = np.asarray(x_eval, dtype=np.float64)
    design_train = np.concatenate([np.ones((x_train.shape[0], 1), dtype=np.float64), x_train], axis=1)
    design_eval = np.concatenate([np.ones((x_eval.shape[0], 1), dtype=np.float64), x_eval], axis=1)
    reg = np.eye(design_train.shape[1], dtype=np.float64) * float(alpha)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(design_train.T @ design_train + reg, design_train.T @ y_train)
    pred = design_eval @ beta
    resid = y_eval - pred
    frac = float(np.linalg.norm(resid) / (np.linalg.norm(y_eval - y_eval.mean(axis=0, keepdims=True)) + 1e-12))
    return resid.astype(np.float32), frac


def crossfit_residualize(y: np.ndarray, x: np.ndarray, *, alpha: float, folds: int, seed: int) -> tuple[np.ndarray, float]:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = int(y.shape[0])
    if n < 4 or int(folds) <= 1:
        return ridge_residualize(y, x, alpha)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_ids = np.zeros(n, dtype=np.int64)
    for fold, idx in enumerate(np.array_split(perm, min(int(folds), n))):
        fold_ids[idx] = fold
    resid = np.zeros_like(y, dtype=np.float64)
    for fold in sorted(np.unique(fold_ids).tolist()):
        test = fold_ids == int(fold)
        train = ~test
        design_train = np.concatenate([np.ones((int(train.sum()), 1)), x[train]], axis=1)
        design_test = np.concatenate([np.ones((int(test.sum()), 1)), x[test]], axis=1)
        reg = np.eye(design_train.shape[1], dtype=np.float64) * float(alpha)
        reg[0, 0] = 0.0
        beta = np.linalg.solve(design_train.T @ design_train + reg, design_train.T @ y[train])
        resid[test] = y[test] - design_test @ beta
    frac = float(np.linalg.norm(resid) / (np.linalg.norm(y - y.mean(axis=0, keepdims=True)) + 1e-12))
    return resid.astype(np.float32), frac


def residualized_feature_sets(
    action_features: Mapping[str, np.ndarray],
    shuffled_features: Mapping[str, np.ndarray],
    static_features: Mapping[str, np.ndarray],
    probe_action_features: Mapping[str, np.ndarray],
    probe_shuffled_features: Mapping[str, np.ndarray],
    probe_static_features: Mapping[str, np.ndarray],
    channels: Sequence[str],
    *,
    alpha: float,
    crossfit_folds: int,
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, object]]]:
    sets = {
        "static_residualized_probefit": {},
        "static_residualized_global": {},
        "static_residualized_crossfit": {},
        "static_residualized_shuffled_probefit": {},
        "static_residualized_shuffled_global": {},
        "static_residualized_shuffled_crossfit": {},
    }
    diagnostics: list[dict[str, object]] = []
    for channel in channels:
        probefit_resid, probefit_frac = ridge_fit_apply(
            probe_action_features[channel],
            probe_static_features[channel],
            action_features[channel],
            static_features[channel],
            alpha,
        )
        global_resid, global_frac = ridge_residualize(action_features[channel], static_features[channel], alpha)
        cross_resid, cross_frac = crossfit_residualize(
            action_features[channel],
            static_features[channel],
            alpha=alpha,
            folds=crossfit_folds,
            seed=seed + 1009,
        )
        shuf_probefit, shuf_probefit_frac = ridge_fit_apply(
            probe_shuffled_features[channel],
            probe_static_features[channel],
            shuffled_features[channel],
            static_features[channel],
            alpha,
        )
        shuf_global, shuf_global_frac = ridge_residualize(shuffled_features[channel], static_features[channel], alpha)
        shuf_cross, shuf_cross_frac = crossfit_residualize(
            shuffled_features[channel],
            static_features[channel],
            alpha=alpha,
            folds=crossfit_folds,
            seed=seed + 2003,
        )
        sets["static_residualized_probefit"][channel] = probefit_resid
        sets["static_residualized_global"][channel] = global_resid
        sets["static_residualized_crossfit"][channel] = cross_resid
        sets["static_residualized_shuffled_probefit"][channel] = shuf_probefit
        sets["static_residualized_shuffled_global"][channel] = shuf_global
        sets["static_residualized_shuffled_crossfit"][channel] = shuf_cross
        diagnostics.extend(
            [
                {"channel": channel, "control": "static_residualized_probefit", "residual_norm_fraction": probefit_frac},
                {"channel": channel, "control": "static_residualized_global", "residual_norm_fraction": global_frac},
                {"channel": channel, "control": "static_residualized_crossfit", "residual_norm_fraction": cross_frac},
                {
                    "channel": channel,
                    "control": "static_residualized_shuffled_probefit",
                    "residual_norm_fraction": shuf_probefit_frac,
                },
                {
                    "channel": channel,
                    "control": "static_residualized_shuffled_global",
                    "residual_norm_fraction": shuf_global_frac,
                },
                {
                    "channel": channel,
                    "control": "static_residualized_shuffled_crossfit",
                    "residual_norm_fraction": shuf_cross_frac,
                },
            ]
        )
    return sets, diagnostics


def conditional_overlap_by_static_bin(
    action_features: Mapping[str, np.ndarray],
    shuffled_features: Mapping[str, np.ndarray],
    static_features: Mapping[str, np.ndarray],
    pair: tuple[str, str],
    *,
    k: int,
    n_bins: int,
) -> list[dict[str, object]]:
    a, b = pair
    static_a = l2_normalize(static_features[a])
    static_b = l2_normalize(static_features[b])
    action_a = l2_normalize(action_features[a])
    action_b = l2_normalize(action_features[b])
    shuffled_a = l2_normalize(shuffled_features[a])
    shuffled_b = l2_normalize(shuffled_features[b])
    n = int(static_a.shape[0])
    static_sim = 0.5 * (static_a @ static_a.T + static_b @ static_b.T)
    action_sim = {a: action_a @ action_a.T, b: action_b @ action_b.T}
    shuffled_sim = {a: shuffled_a @ shuffled_a.T, b: shuffled_b @ shuffled_b.T}
    rows: list[dict[str, object]] = []
    for bin_id in range(int(n_bins)):
        control_scores = {"action_effect_static_conditioned": [], "shuffled_static_conditioned": []}
        control_chances = {"action_effect_static_conditioned": [], "shuffled_static_conditioned": []}
        pool_sizes = []
        for i in range(n):
            candidates = np.asarray([j for j in range(n) if j != i], dtype=np.int64)
            if candidates.size == 0:
                continue
            order = candidates[np.argsort(-static_sim[i, candidates], kind="mergesort")]
            chunks = np.array_split(order, int(n_bins))
            pool = chunks[bin_id]
            if pool.size < 2:
                continue
            kk = min(int(k), int(pool.size))
            pool_sizes.append(int(pool.size))
            for control, sims in [
                ("action_effect_static_conditioned", action_sim),
                ("shuffled_static_conditioned", shuffled_sim),
            ]:
                top_a = pool[np.argsort(-sims[a][i, pool], kind="mergesort")[:kk]]
                top_b = pool[np.argsort(-sims[b][i, pool], kind="mergesort")[:kk]]
                score = len(set(top_a.tolist()) & set(top_b.tolist())) / float(kk)
                control_scores[control].append(float(score))
                control_chances[control].append(float(kk / max(1, pool.size)))
        for control in ["action_effect_static_conditioned", "shuffled_static_conditioned"]:
            scores = np.asarray(control_scores[control], dtype=np.float64)
            chances = np.asarray(control_chances[control], dtype=np.float64)
            overlap = float(scores.mean()) if scores.size else float("nan")
            chance = float(chances.mean()) if chances.size else float("nan")
            rows.append(
                {
                    "control": control,
                    "static_bin": int(bin_id),
                    "static_bin_label": ["highest", "high_mid", "low_mid", "lowest"][bin_id]
                    if int(n_bins) == 4
                    else f"bin_{bin_id}",
                    "n_queries": int(scores.size),
                    "mean_candidate_pool": float(np.mean(pool_sizes)) if pool_sizes else float("nan"),
                    "k": int(k),
                    "overlap": overlap,
                    "random_expected_overlap": chance,
                    "chance_adjusted_overlap": overlap - chance if np.isfinite(overlap) else float("nan"),
                }
            )
    return rows


def discover_runs(root: Path) -> list[Path]:
    return sorted({p.parents[2] for p in root.glob("seed_*/split_*/pca_*/stageb6_diagnostics/reports/stageb6_summary.json")})


def filter_runs(runs: Sequence[Path], args: argparse.Namespace) -> list[Path]:
    out = []
    pca_dims = None if args.pca_dims is None else {int(x) for x in args.pca_dims}
    data_seeds = None if args.data_seeds is None else {int(x) for x in args.data_seeds}
    split_seeds = None if args.split_seeds is None else {int(x) for x in args.split_seeds}
    for run in runs:
        data_seed = int(run.parts[-3].removeprefix("seed_"))
        split_seed = int(run.parts[-2].removeprefix("split_"))
        pca_dim = int(run.parts[-1].removeprefix("pca_"))
        if pca_dims is not None and pca_dim not in pca_dims:
            continue
        if data_seeds is not None and data_seed not in data_seeds:
            continue
        if split_seeds is not None and split_seed not in split_seeds:
            continue
        out.append(run)
    return sorted(out)


def analyze_run(
    run: Path,
    root: Path,
    args: argparse.Namespace,
    channels_arg: Sequence[str] | None,
    pairs: Sequence[tuple[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    data_seed = int(run.parts[-3].removeprefix("seed_"))
    split_seed = int(run.parts[-2].removeprefix("split_"))
    pca_dim = int(run.parts[-1].removeprefix("pca_"))
    data_path = root / f"seed_{data_seed}" / "stage0" / "stage0_dataset.npz"
    model_path = run / "action_effect_probe" / "transition_encoders.npz"
    static_model_path = run / "static" / "transition_encoders.npz"
    split_path = run / "split" / "probe_action_ids.txt"
    header = np.load(data_path, allow_pickle=True)
    all_channels = [str(x) for x in np.asarray(header["channels"]).astype(str).tolist()]
    blocked: set[str] = set()
    if "diagnostic_only_channels" in header.files:
        blocked = {str(x) for x in np.asarray(header["diagnostic_only_channels"]).astype(str).tolist()}
    header.close()
    channels = [ch for ch in (channels_arg or all_channels) if ch in all_channels and ch not in blocked]
    data = load_static_control_npz(data_path, channels)
    encoders, _ = load_transition_encoders(str(model_path))
    static_encoders, _ = load_transition_encoders(str(static_model_path))
    sample_indices = validate_dense_stage0_data(data, channels)
    static_sample_indices = validate_dense_static_data(data, channels)
    states, action_ids, dense_rows, full_rows = complete_state_action_matrix(
        data,
        sample_indices,
        max_states=int(args.max_states),
        seed=int(args.seed) + data_seed,
    )
    probe_set = set(read_probe_ids(split_path))
    probe_pos = np.asarray([i for i, aid in enumerate(action_ids.tolist()) if int(aid) in probe_set], dtype=np.int64)
    heldout_pos = np.asarray([i for i, aid in enumerate(action_ids.tolist()) if int(aid) not in probe_set], dtype=np.int64)
    full_action_types = validate_action_type_columns(np.asarray(data["action_type"])[full_rows], action_ids)
    static_dense_rows = dense_rows_for_sample_ids(static_sample_indices, full_rows)
    tensors = embedding_tensors_from_model(data, encoders, channels, prefix="delta", dense_rows=dense_rows)
    static_tensors = embedding_tensors_from_model(data, static_encoders, channels, prefix="obs0", dense_rows=static_dense_rows)
    meta = run_metadata(run, root)
    meta.update({"n_states": int(states.shape[0]), "n_actions": int(action_ids.shape[0]), "n_heldout_actions": int(heldout_pos.size)})

    residual_rows: list[dict[str, object]] = []
    conditioned_rows: list[dict[str, object]] = []
    literature_rows: list[dict[str, object]] = []
    diag_rows: list[dict[str, object]] = []

    for norm in [str(x) for x in args.normalization_modes]:
        if norm not in NORMALIZATION_MODES:
            raise ValueError(f"Unsupported normalization mode: {norm}")
        ref_pos = probe_pos if norm in {"probe_global_apply", "probe_action_type_apply"} else None
        probe_action_features = features_from_tensors(
            tensors,
            channels,
            probe_pos,
            mode=norm,
            action_types=full_action_types,
            reference_positions=ref_pos,
        )
        action_features = features_from_tensors(
            tensors,
            channels,
            heldout_pos,
            mode=norm,
            action_types=full_action_types,
            reference_positions=ref_pos,
        )
        probe_static_features = features_from_tensors(
            static_tensors,
            channels,
            probe_pos,
            mode=norm,
            action_types=full_action_types,
            reference_positions=ref_pos,
        )
        static_features = features_from_tensors(
            static_tensors,
            channels,
            heldout_pos,
            mode=norm,
            action_types=full_action_types,
            reference_positions=ref_pos,
        )
        probe_shuffled_features = normalized_then_shuffled_features_from_tensors(
            tensors,
            channels,
            probe_pos,
            mode=norm,
            action_types=full_action_types,
            reference_positions=ref_pos,
            seed=int(args.seed) + data_seed * 1009 + split_seed + 11,
        )
        shuffled_features = normalized_then_shuffled_features_from_tensors(
            tensors,
            channels,
            heldout_pos,
            mode=norm,
            action_types=full_action_types,
            reference_positions=ref_pos,
            seed=int(args.seed) + data_seed * 1009 + split_seed,
        )
        residual_sets, diagnostics = residualized_feature_sets(
            action_features,
            shuffled_features,
            static_features,
            probe_action_features,
            probe_shuffled_features,
            probe_static_features,
            channels,
            alpha=float(args.residual_ridge_alpha),
            crossfit_folds=int(args.crossfit_folds),
            seed=int(args.seed) + data_seed * 1009 + split_seed,
        )
        for row in diagnostics:
            diag_rows.append({**meta, "normalization": norm, "residual_ridge_alpha": float(args.residual_ridge_alpha), **row})
        for k in [int(x) for x in args.k_values]:
            for pair in pairs:
                if pair[0] not in channels or pair[1] not in channels:
                    continue
                for control, feats in residual_sets.items():
                    base, scores = overlap_summary(feats, pair, k, int(args.batch_size))
                    residual_rows.append(
                        {
                            **meta,
                            "normalization": norm,
                            "channel_a": pair[0],
                            "channel_b": pair[1],
                            "control": control,
                            "residual_primary": bool(control == "static_residualized_probefit"),
                            "residual_ridge_alpha": float(args.residual_ridge_alpha),
                            "crossfit_folds": int(args.crossfit_folds),
                            **base,
                            "per_query_score_mean": float(np.nanmean(scores)) if scores.size else float("nan"),
                        }
                    )
                conditioned_rows.extend(
                    {
                        **meta,
                        "normalization": norm,
                        "channel_a": pair[0],
                        "channel_b": pair[1],
                        **row,
                    }
                    for row in conditional_overlap_by_static_bin(
                        action_features,
                        shuffled_features,
                        static_features,
                        pair,
                        k=k,
                        n_bins=int(args.static_bins),
                    )
                )
            if bool(args.skip_literature_metrics) or int(k) != int(args.literature_k):
                continue
            for pair in pairs:
                if pair[0] not in channels or pair[1] not in channels:
                    continue
                for control, feats in residual_sets.items():
                    x = feats[pair[0]]
                    y = feats[pair[1]]
                    cycle_value, cycle_cal = calibrated_cycle_knn(
                        x,
                        y,
                        k=int(args.literature_k),
                        repeats=int(args.permutation_repeats),
                        seed=int(args.seed) + data_seed * 100003 + split_seed * 1009 + pca_dim,
                    )
                    cknna_value, cknna_cal = calibrated_cknna(
                        x,
                        y,
                        k=int(args.literature_k),
                        repeats=int(args.permutation_repeats),
                        seed=int(args.seed) + data_seed * 100003 + split_seed * 1009 + pca_dim + 17,
                    )
                    for measurement, value, calibration in [
                        ("cycle_knn_overlap", cycle_value, cycle_cal),
                        ("cknna_linear_cka", cknna_value, cknna_cal),
                    ]:
                        literature_rows.append(
                            {
                                **meta,
                                "normalization": norm,
                                "channel_a": pair[0],
                                "channel_b": pair[1],
                                "control": control,
                                "measurement": measurement,
                                "value": float(value),
                                "k": int(args.literature_k),
                                **{f"null_{key}": val for key, val in calibration.items()},
                            }
                        )
    return residual_rows, conditioned_rows, literature_rows, diag_rows


def summarize(
    residual_rows: list[dict[str, object]],
    conditioned_rows: list[dict[str, object]],
    literature_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    primary = [
        row
        for row in residual_rows
        if str(row.get("pca_dim")) == "32"
        and row.get("normalization") == "probe_action_type_apply"
        and row.get("channel_a") == "rgb"
        and row.get("channel_b") == "range"
        and int(row.get("k", 0)) == 10
    ]
    for control in [
        "static_residualized_probefit",
        "static_residualized_shuffled_probefit",
        "static_residualized_crossfit",
        "static_residualized_shuffled_crossfit",
        "static_residualized_global",
        "static_residualized_shuffled_global",
    ]:
        vals = np.asarray([float(row["chance_adjusted_overlap"]) for row in primary if row["control"] == control], dtype=np.float64)
        rows.append(
            {
                "summary": "primary_residualized",
                "control": control,
                "n": int(vals.size),
                "chance_adjusted_mean": float(vals.mean()) if vals.size else float("nan"),
                "chance_adjusted_min": float(vals.min()) if vals.size else float("nan"),
                "positive_count": int(np.sum(vals > 0.0)) if vals.size else 0,
            }
        )
    cond = [
        row
        for row in conditioned_rows
        if str(row.get("pca_dim")) == "32"
        and row.get("normalization") == "probe_action_type_apply"
        and row.get("channel_a") == "rgb"
        and row.get("channel_b") == "range"
        and int(row.get("k", 0)) == 10
    ]
    for control in ["action_effect_static_conditioned", "shuffled_static_conditioned"]:
        for bin_label in sorted({str(row["static_bin_label"]) for row in cond}):
            vals = np.asarray(
                [float(row["chance_adjusted_overlap"]) for row in cond if row["control"] == control and str(row["static_bin_label"]) == bin_label],
                dtype=np.float64,
            )
            rows.append(
                {
                    "summary": "primary_static_conditioned",
                    "control": control,
                    "static_bin_label": bin_label,
                    "n": int(vals.size),
                    "chance_adjusted_mean": float(vals.mean()) if vals.size else float("nan"),
                    "chance_adjusted_min": float(vals.min()) if vals.size else float("nan"),
                    "positive_count": int(np.sum(vals > 0.0)) if vals.size else 0,
                }
            )
    lit = [
        row
        for row in literature_rows
        if str(row.get("pca_dim")) == "32"
        and row.get("normalization") == "probe_action_type_apply"
        and row.get("channel_a") == "rgb"
        and row.get("channel_b") == "range"
    ]
    for measurement in ["cycle_knn_overlap", "cknna_linear_cka"]:
        for control in ["static_residualized_probefit", "static_residualized_shuffled_probefit", "static_residualized_crossfit", "static_residualized_shuffled_crossfit"]:
            vals = np.asarray(
                [float(row["null_calibrated_score"]) for row in lit if row["measurement"] == measurement and row["control"] == control],
                dtype=np.float64,
            )
            rows.append(
                {
                    "summary": "primary_literature_metric",
                    "measurement": measurement,
                    "control": control,
                    "n": int(vals.size),
                    "calibrated_mean": float(vals.mean()) if vals.size else float("nan"),
                    "calibrated_min": float(vals.min()) if vals.size else float("nan"),
                    "positive_count": int(np.sum(vals > 0.0)) if vals.size else 0,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root
    all_runs = discover_runs(root)
    runs = filter_runs(all_runs, args)
    if args.expected_report_dirs is not None and len(runs) != int(args.expected_report_dirs):
        raise RuntimeError(f"Expected {args.expected_report_dirs} completed B6 runs, found {len(runs)}")
    pairs = parse_pairs(args.target_pairs)
    channels = [str(x) for x in args.channels] if args.channels else None
    residual_rows: list[dict[str, object]] = []
    conditioned_rows: list[dict[str, object]] = []
    literature_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for run in runs:
        r, c, l, d = analyze_run(run, root, args, channels, pairs)
        residual_rows.extend(r)
        conditioned_rows.extend(c)
        literature_rows.extend(l)
        diagnostic_rows.extend(d)
    write_csv(out / "stageb6_static_residualized_knn.csv", residual_rows)
    write_csv(out / "stageb6_static_conditioned_knn.csv", conditioned_rows)
    write_csv(out / "stageb6_static_control_literature_metrics.csv", literature_rows)
    write_csv(out / "stageb6_static_residual_diagnostics.csv", diagnostic_rows)
    summary_rows = summarize(residual_rows, conditioned_rows, literature_rows)
    write_csv(out / "stageb6_static_control_summary.csv", summary_rows)
    summary = {
        "root": str(root),
        "n_runs": int(len(runs)),
        "n_residual_rows": int(len(residual_rows)),
        "n_conditioned_rows": int(len(conditioned_rows)),
        "n_literature_rows": int(len(literature_rows)),
        "primary_summary": summary_rows,
        "decision_note": (
            "Static-control gate passes only if residualized action-effect clearly beats "
            "residualized shuffled and conditioned action-effect remains positive in "
            "static-similarity bins."
        ),
    }
    ensure_dir(out)
    with open(out / "stageb6_static_control_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, allow_nan=True)
    print(f"Wrote Stage B.6 static-control reports to {out}")


if __name__ == "__main__":
    main()
