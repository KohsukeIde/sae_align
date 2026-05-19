#!/usr/bin/env python
"""Stage B.6 artifact checks and measurement-primitive sanity diagnostics.

B.6 keeps the B.5 held-out same-action-set setup, but asks whether the weak
`rgb-range` signal is stable under k, jitter, PCA basis, and measurement changes.
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
    corr,
    features_from_tensors,
    normalize_tensor,
    normalization_is_primary,
    normalization_is_transductive,
    score_maps,
)
from analyze_stageb4_reliability import (
    CORE_REDUNDANCY_PAIRS,
    DEFAULT_REDUNDANCY_PAIRS,
    bootstrap_paired_diff_ci,
    feature_diagnostic_rows,
)
from analyze_stageb5_heldout_alignment import (
    DEFAULT_CHANNELS,
    TARGET_PAIRS,
    bootstrap_row as bootstrap_overlap_row,
    embedding_tensors_from_model,
    mean_finite,
    neighbor_sets,
    overlap_row,
    per_query_scores,
    random_project_tensors,
    raw_tensors,
)
from analyze_state_signature_knn import (
    complete_state_action_matrix,
    dense_rows_for_sample_ids,
    load_npz,
    state_pair_strata,
    validate_model_data_path,
    validate_model_fingerprint,
    validate_probe_trained_model,
)
from sae_align.analysis.strata import diagnostic_only_channels, validate_dense_stage0_data, validate_dense_static_data
from sae_align.models import load_transition_encoders
from sae_align.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model", type=str, required=True, help="Probe-action-only action-effect PCA model.")
    p.add_argument("--static-model", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--all-action-model", type=str, default=None)
    p.add_argument("--channels", nargs="*", default=list(DEFAULT_CHANNELS))
    p.add_argument("--probe-action-ids", nargs="*", type=int, required=True)
    p.add_argument(
        "--representations",
        nargs="*",
        default=["pca_probe_only", "raw_delta", "random_projection"],
    )
    p.add_argument("--random-projection-dim", type=int, default=32)
    p.add_argument("--random-projection-seed", type=int, default=0)
    p.add_argument("--normalization-modes", nargs="*", default=["none", "probe_global_apply", "probe_action_type_apply"])
    p.add_argument("--k-values", nargs="*", type=int, default=[5, 10, 20])
    p.add_argument("--jitter-epsilons", nargs="*", type=float, default=[0.0, 1e-6, 1e-5, 1e-4, 1e-3])
    p.add_argument("--jitter-seeds", nargs="*", type=int, default=list(range(10)))
    p.add_argument("--target-pairs", nargs="*", default=["rgb:range"])
    p.add_argument("--max-states", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--detectability-quantile", type=float, default=0.10)
    p.add_argument("--regular-state-threshold", type=float, default=0.25)
    p.add_argument("--blind-state-threshold", type=float, default=0.25)
    p.add_argument("--physical-state-threshold", type=float, default=0.25)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--ridge-splits", type=int, default=10)
    p.add_argument("--sanity-knn-k", type=int, default=10)
    p.add_argument("--svcca-components", type=int, default=32)
    p.add_argument("--extended-metric-repeats", type=int, default=50)
    p.add_argument("--permutation-repeats", type=int, default=200)
    p.add_argument("--permutation-seed", type=int, default=0)
    p.add_argument("--bootstrap-repeats", type=int, default=200)
    p.add_argument("--bootstrap-seed", type=int, default=0)
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


def parse_pairs(values: Sequence[str]) -> list[tuple[str, str]]:
    pairs = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"Pairs must be formatted channel_a:channel_b, got {value!r}")
        a, b = value.split(":", 1)
        pairs.append((a, b))
    return pairs


def pca_component_count(encoders: Mapping[str, object]) -> int:
    dims = []
    for encoder in encoders.values():
        components = getattr(encoder, "components_", None)
        if components is not None:
            dims.append(int(components.shape[0]))
    return int(min(dims)) if dims else 0


def jittered_features(features: Mapping[str, np.ndarray], *, epsilon: float, seed: int) -> dict[str, np.ndarray]:
    if float(epsilon) == 0.0:
        return {k: np.asarray(v, dtype=np.float32) for k, v in features.items()}
    rng = np.random.default_rng(int(seed))
    out = {}
    for channel, x in features.items():
        x = np.asarray(x, dtype=np.float32)
        noise = rng.normal(0.0, float(epsilon), size=x.shape).astype(np.float32)
        out[channel] = (x + noise).astype(np.float32)
    return out


def action_type_shuffled_features_from_tensors(
    tensors: Mapping[str, np.ndarray],
    channels: Sequence[str],
    positions: np.ndarray,
    *,
    mode: str,
    action_types: np.ndarray,
    reference_positions: np.ndarray | None = None,
    seed: int,
) -> dict[str, np.ndarray]:
    """Shuffle held-out action columns within action type for each state.

    This preserves type-specific scale and the action-type normalization path,
    while breaking exact held-out action identity where multiple held-out
    actions of the same type are available.
    """
    rng = np.random.default_rng(seed)
    out = {}
    heldout_types = action_types[positions].astype(str)
    for channel in channels:
        tensor = tensors[channel][:, positions, :].copy()
        for state_i in range(tensor.shape[0]):
            for label in np.unique(heldout_types):
                cols = np.flatnonzero(heldout_types == str(label))
                if cols.size > 1:
                    tensor[state_i, cols, :] = tensor[state_i, rng.permutation(cols), :]
        reference = None if reference_positions is None else tensors[channel][:, reference_positions, :]
        reference_types = None if reference_positions is None else action_types[reference_positions]
        normed = normalize_tensor(
            tensor,
            mode=mode,
            action_types=heldout_types,
            reference=reference,
            reference_action_types=reference_types,
        )
        out[channel] = normed.reshape(normed.shape[0], -1).astype(np.float32)
    return out


def normalized_then_shuffled_features_from_tensors(
    tensors: Mapping[str, np.ndarray],
    channels: Sequence[str],
    positions: np.ndarray,
    *,
    mode: str,
    action_types: np.ndarray,
    reference_positions: np.ndarray | None = None,
    seed: int,
) -> dict[str, np.ndarray]:
    """Normalize held-out columns, then shuffle action columns within state.

    The B.3/B.5 shuffled helper shuffled before normalization. That is
    intentionally destructive under action-type normalization because the
    shuffled columns no longer match their action-type labels. B.6 uses this
    safer null for the primary shuffled control.
    """
    rng = np.random.default_rng(seed)
    out = {}
    heldout_types = action_types[positions].astype(str)
    for channel in channels:
        tensor = tensors[channel]
        reference = None if reference_positions is None else tensor[:, reference_positions, :]
        reference_types = None if reference_positions is None else action_types[reference_positions]
        normed = normalize_tensor(
            tensor[:, positions, :],
            mode=mode,
            action_types=heldout_types,
            reference=reference,
            reference_action_types=reference_types,
        )
        for state_i in range(normed.shape[0]):
            normed[state_i] = normed[state_i, rng.permutation(normed.shape[1]), :]
        out[channel] = normed.reshape(normed.shape[0], -1).astype(np.float32)
    return out


def representation_flags(representation: str) -> dict[str, object]:
    rep = str(representation)
    return {
        "representation_primary": bool(rep == "pca_probe_only"),
        "representation_diagnostic": bool(rep != "pca_probe_only"),
        "pca_basis_source": {
            "pca_probe_only": "probe_actions_only",
            "pca_all_action": "all_actions_leaky_upper_bound",
            "raw_delta": "none",
            "random_projection": "random_projection",
        }.get(rep, "unknown"),
        "uses_heldout_actions_in_pca_basis": bool(rep == "pca_all_action"),
        "leakage_diagnostic": bool(rep == "pca_all_action"),
        "primary_evidence_eligible": bool(rep == "pca_probe_only"),
    }


def validate_action_type_columns(action_types: np.ndarray, action_ids: np.ndarray) -> np.ndarray:
    action_types = np.asarray(action_types).astype(str)
    action_ids = np.asarray(action_ids)
    if action_types.ndim != 2:
        raise ValueError("Expected dense action_type matrix with shape states x actions.")
    out = []
    for pos in range(action_types.shape[1]):
        values = np.unique(action_types[:, pos])
        if values.size != 1:
            raise ValueError(
                "Stage B.6 action-type normalization requires action_type to be constant "
                f"for action_id={int(action_ids[pos])}; got {values.tolist()}"
            )
        out.append(str(values[0]))
    return np.asarray(out, dtype=str)


def validate_all_action_model(metadata: Mapping[str, object], expected_action_ids: Sequence[int]) -> str:
    train_action_ids = metadata.get("actual_train_action_ids", metadata.get("train_action_ids"))
    if train_action_ids is None:
        return "all_or_unspecified"
    train_set = {int(x) for x in train_action_ids}
    expected = {int(x) for x in expected_action_ids}
    if train_set != expected:
        raise ValueError(
            "pca_all_action diagnostic must be fitted on the full action bank. "
            f"Got train actions {sorted(train_set)}, expected {sorted(expected)}."
        )
    return "all_actions"


def center(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=0, keepdims=True)


def average_rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sorted_x = x[order]
    start = 0
    while start < x.size:
        end = start + 1
        while end < x.size and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    denom = max(1.0, float(x.size - 1))
    return ranks / denom


def corr_average_rank(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan"), float("nan"), int(x.size)
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx = average_rank01(x)
    ry = average_rank01(y)
    if float(np.std(rx)) <= 1e-12 or float(np.std(ry)) <= 1e-12:
        return pearson, float("nan"), int(x.size)
    return pearson, float(np.corrcoef(rx, ry)[0, 1]), int(x.size)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must have the same number of rows.")
    kx = centered_linear_gram(x)
    ky = centered_linear_gram(y)
    return linear_cka_from_grams(kx, ky)


def centered_linear_gram(x: np.ndarray) -> np.ndarray:
    x = center(x)
    return (x @ x.T).astype(np.float64)


def linear_cka_from_grams(kx: np.ndarray, ky: np.ndarray) -> float:
    denom = np.sqrt(float(np.sum(kx * kx)) * float(np.sum(ky * ky)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(kx * ky) / denom)


def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norm, 1e-12)
    return x @ x.T


def directed_knn_adjacency(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = int(x.shape[0])
    kk = max(0, min(int(k), n - 1))
    adj = np.zeros((n, n), dtype=bool)
    if n <= 1 or kk == 0:
        return adj
    sim = cosine_similarity_matrix(x)
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1, kind="mergesort")[:, :kk]
    rows = np.repeat(np.arange(n), kk)
    adj[rows, order.reshape(-1)] = True
    return adj


def cycle_knn_overlap_from_adjacency(ax: np.ndarray, ay: np.ndarray) -> float:
    cx = np.asarray(ax, dtype=bool) & np.asarray(ax, dtype=bool).T
    cy = np.asarray(ay, dtype=bool) & np.asarray(ay, dtype=bool).T
    denom = np.maximum(cx.sum(axis=1), 1)
    values = (cx & cy).sum(axis=1) / denom
    return float(np.mean(values)) if values.size else float("nan")


def cknna_from_adjacency(ax: np.ndarray, ay: np.ndarray) -> float:
    gx = (np.asarray(ax, dtype=bool) | np.asarray(ax, dtype=bool).T).astype(np.float64)
    gy = (np.asarray(ay, dtype=bool) | np.asarray(ay, dtype=bool).T).astype(np.float64)
    return linear_cka_from_grams(center_graph(gx), center_graph(gy))


def center_graph(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + float(a.mean())


def calibrated_cycle_knn(
    x: np.ndarray,
    y: np.ndarray,
    *,
    k: int,
    repeats: int,
    seed: int,
) -> tuple[float, dict[str, object]]:
    ax = directed_knn_adjacency(x, k=int(k))
    ay = directed_knn_adjacency(y, k=int(k))
    observed = cycle_knn_overlap_from_adjacency(ax, ay)
    rng = np.random.default_rng(int(seed))
    null = []
    for _ in range(int(repeats)):
        perm = rng.permutation(ay.shape[0])
        null.append(cycle_knn_overlap_from_adjacency(ax, ay[np.ix_(perm, perm)]))
    return observed, calibration_fields(observed, null)


def calibrated_cknna(
    x: np.ndarray,
    y: np.ndarray,
    *,
    k: int,
    repeats: int,
    seed: int,
) -> tuple[float, dict[str, object]]:
    ax = directed_knn_adjacency(x, k=int(k))
    ay = directed_knn_adjacency(y, k=int(k))
    observed = cknna_from_adjacency(ax, ay)
    rng = np.random.default_rng(int(seed))
    null = []
    for _ in range(int(repeats)):
        perm = rng.permutation(ay.shape[0])
        null.append(cknna_from_adjacency(ax, ay[np.ix_(perm, perm)]))
    return observed, calibration_fields(observed, null)


def canonical_correlations(
    x: np.ndarray,
    y: np.ndarray,
    *,
    components: int,
    eps: float = 1e-7,
) -> np.ndarray:
    x = center(np.asarray(x, dtype=np.float64))
    y = center(np.asarray(y, dtype=np.float64))
    if x.shape[0] != y.shape[0] or x.shape[0] < 3:
        return np.asarray([], dtype=np.float64)
    ux, sx, _ = np.linalg.svd(x, full_matrices=False)
    uy, sy, _ = np.linalg.svd(y, full_matrices=False)
    if sx.size == 0 or sy.size == 0:
        return np.asarray([], dtype=np.float64)
    keep_x = sx > float(eps) * max(float(sx[0]), 1.0)
    keep_y = sy > float(eps) * max(float(sy[0]), 1.0)
    max_rank = max(1, int(components))
    ux = ux[:, keep_x][:, :max_rank]
    uy = uy[:, keep_y][:, :max_rank]
    if ux.shape[1] == 0 or uy.shape[1] == 0:
        return np.asarray([], dtype=np.float64)
    corr_values = np.linalg.svd(ux.T @ uy, compute_uv=False)
    return np.clip(np.asarray(corr_values, dtype=np.float64), 0.0, 1.0)


def calibrated_svcca_mean(
    x: np.ndarray,
    y: np.ndarray,
    *,
    components: int,
    repeats: int,
    seed: int,
) -> tuple[float, int, dict[str, object]]:
    values = canonical_correlations(x, y, components=int(components))
    observed = float(values.mean()) if values.size else float("nan")
    rng = np.random.default_rng(int(seed))
    null = []
    for _ in range(int(repeats)):
        perm = rng.permutation(y.shape[0])
        shuffled = canonical_correlations(x, y[perm], components=int(components))
        null.append(float(shuffled.mean()) if shuffled.size else float("nan"))
    return observed, int(values.size), calibration_fields(observed, null)


def rsa_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    if x.shape[0] != y.shape[0] or x.shape[0] < 3:
        return float("nan"), int(min(x.shape[0], y.shape[0]))
    return rsa_spearman_from_similarity(cosine_similarity_matrix(x), cosine_similarity_matrix(y))


def rsa_spearman_from_similarity(sx: np.ndarray, sy: np.ndarray) -> tuple[float, int]:
    if sx.shape != sy.shape or sx.shape[0] < 3:
        return float("nan"), int(min(sx.shape[0], sy.shape[0]))
    tri = np.triu_indices(sx.shape[0], k=1)
    _, spearman, n = corr_average_rank(sx[tri], sy[tri])
    return spearman, n


def ridge_r2_cv(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    splits: int,
    seed: int,
) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] != y.shape[0] or x.shape[0] < 4:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    scores = []
    n = x.shape[0]
    test_n = max(1, int(round(n * 0.25)))
    for _ in range(int(splits)):
        perm = rng.permutation(n)
        test = perm[:test_n]
        train = perm[test_n:]
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]
        x_mean = x_train.mean(axis=0, keepdims=True)
        x_std = x_train.std(axis=0, keepdims=True) + 1e-6
        y_mean = y_train.mean(axis=0, keepdims=True)
        xs = (x_train - x_mean) / x_std
        xt = (x_test - x_mean) / x_std
        yc = y_train - y_mean
        if xs.shape[1] <= xs.shape[0]:
            gram = xs.T @ xs
            coef = np.linalg.solve(gram + float(alpha) * np.eye(gram.shape[0]), xs.T @ yc)
            pred = xt @ coef + y_mean
        else:
            gram = xs @ xs.T
            dual_coef = np.linalg.solve(gram + float(alpha) * np.eye(gram.shape[0]), yc)
            pred = (xt @ xs.T) @ dual_coef + y_mean
        ss_res = float(np.sum((y_test - pred) ** 2))
        ss_tot = float(np.sum((y_test - y_test.mean(axis=0, keepdims=True)) ** 2))
        scores.append(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan"))
    clean = np.asarray(scores, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan")
    return float(clean.mean()), float(clean.std(ddof=0))


def calibration_fields(observed: float, null_values: Sequence[float]) -> dict[str, object]:
    null = np.asarray(null_values, dtype=np.float64)
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or null.size == 0:
        return {
            "null_repeats": int(null.size),
            "null_mean": float("nan"),
            "null_q95": float("nan"),
            "p_add_one": float("nan"),
            "calibrated_score": float("nan"),
        }
    p = (1.0 + float(np.sum(null >= float(observed)))) / float(null.size + 1)
    return {
        "null_repeats": int(null.size),
        "null_mean": float(null.mean()),
        "null_q95": float(np.quantile(null, 0.95)),
        "p_add_one": p,
        "calibrated_score": float(observed) - float(null.mean()),
    }


def calibrated_linear_cka(
    x: np.ndarray,
    y: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, dict[str, object]]:
    kx = centered_linear_gram(x)
    ky = centered_linear_gram(y)
    observed = linear_cka_from_grams(kx, ky)
    rng = np.random.default_rng(int(seed))
    null = []
    for _ in range(int(repeats)):
        perm = rng.permutation(y.shape[0])
        null.append(linear_cka_from_grams(kx, ky[np.ix_(perm, perm)]))
    return observed, calibration_fields(observed, null)


def calibrated_rsa_spearman(
    x: np.ndarray,
    y: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> tuple[float, int, dict[str, object]]:
    sx = cosine_similarity_matrix(x)
    sy = cosine_similarity_matrix(y)
    observed, n = rsa_spearman_from_similarity(sx, sy)
    rng = np.random.default_rng(int(seed))
    null = []
    for _ in range(int(repeats)):
        perm = rng.permutation(y.shape[0])
        null.append(rsa_spearman_from_similarity(sx, sy[np.ix_(perm, perm)])[0])
    return observed, n, calibration_fields(observed, null)


def calibrated_ridge_r2(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    splits: int,
    repeats: int,
    seed: int,
) -> tuple[float, float, dict[str, object]]:
    observed, observed_std = ridge_r2_cv(x, y, alpha=float(alpha), splits=int(splits), seed=int(seed))
    rng = np.random.default_rng(int(seed) + 17)
    null = [
        ridge_r2_cv(
            x,
            y[rng.permutation(y.shape[0])],
            alpha=float(alpha),
            splits=int(splits),
            seed=int(seed) + 1009 + i,
        )[0]
        for i in range(int(repeats))
    ]
    return observed, observed_std, calibration_fields(observed, null)


def sanity_rows_for_features(
    *,
    representation: str,
    representation_components: int,
    control: str,
    normalization: str,
    features: Mapping[str, np.ndarray],
    tensors: Mapping[str, np.ndarray] | None,
    positions: np.ndarray,
    action_types: np.ndarray,
    reference_positions: np.ndarray | None,
    pairs: Sequence[tuple[str, str]],
    ridge_alpha: float,
    ridge_splits: int,
    sanity_knn_k: int,
    svcca_components: int,
    extended_metric_repeats: int,
    permutation_repeats: int,
    seed: int,
    control_feature_source: str,
) -> list[dict[str, object]]:
    rows = []
    normed_tensors: dict[str, np.ndarray] | None = None
    if tensors is not None:
        normed_tensors = {}
        for channel, tensor in tensors.items():
            reference = None if reference_positions is None else tensor[:, reference_positions, :]
            reference_types = None if reference_positions is None else action_types[reference_positions]
            normed_tensors[channel] = normalize_tensor(
                tensor[:, positions, :],
                mode=normalization,
                action_types=action_types[positions],
                reference=reference,
                reference_action_types=reference_types,
            )
    for a, b in pairs:
        if a not in features or b not in features:
            continue
        x = features[a]
        y = features[b]
        cka_value, cka_calibration = calibrated_linear_cka(
            x,
            y,
            repeats=int(permutation_repeats),
            seed=int(seed) + 101,
        )
        rsa_value, rsa_n, rsa_calibration = calibrated_rsa_spearman(
            x,
            y,
            repeats=int(permutation_repeats),
            seed=int(seed) + 151,
        )
        r2, r2_std, r2_calibration = calibrated_ridge_r2(
            x,
            y,
            alpha=float(ridge_alpha),
            splits=int(ridge_splits),
            repeats=int(permutation_repeats),
            seed=int(seed) + 202,
        )
        r2_reverse, r2_reverse_std, r2_reverse_calibration = calibrated_ridge_r2(
            y,
            x,
            alpha=float(ridge_alpha),
            splits=int(ridge_splits),
            repeats=int(permutation_repeats),
            seed=int(seed) + 203,
        )
        rows.append(
            {
                "representation": representation,
                "representation_components": int(representation_components),
                **representation_flags(representation),
                "control": control,
                "control_feature_source": control_feature_source,
                "normalization": normalization,
                "channel_a": a,
                "channel_b": b,
                "channel_a_dim": int(x.shape[1]),
                "channel_b_dim": int(y.shape[1]),
                "measurement": "state_flat_linear_cka",
                "value": cka_value,
                "std": float("nan"),
                "n": int(x.shape[0]),
                "sanity_knn_k": int(sanity_knn_k),
                "svcca_components": int(svcca_components),
                "extended_metric_repeats": int(extended_metric_repeats),
                "ridge_alpha": float(ridge_alpha),
                "ridge_splits": int(ridge_splits),
                "permutation_repeats": int(permutation_repeats),
                **cka_calibration,
            }
        )
        rows.append(
            {
                "representation": representation,
                "representation_components": int(representation_components),
                **representation_flags(representation),
                "control": control,
                "control_feature_source": control_feature_source,
                "normalization": normalization,
                "channel_a": a,
                "channel_b": b,
                "channel_a_dim": int(x.shape[1]),
                "channel_b_dim": int(y.shape[1]),
                "measurement": "state_flat_rsa_spearman",
                "value": rsa_value,
                "std": float("nan"),
                "n": int(rsa_n),
                "sanity_knn_k": int(sanity_knn_k),
                "svcca_components": int(svcca_components),
                "extended_metric_repeats": int(extended_metric_repeats),
                "ridge_alpha": float(ridge_alpha),
                "ridge_splits": int(ridge_splits),
                "permutation_repeats": int(permutation_repeats),
                **rsa_calibration,
            }
        )
        rows.append(
            {
                "representation": representation,
                "representation_components": int(representation_components),
                **representation_flags(representation),
                "control": control,
                "control_feature_source": control_feature_source,
                "normalization": normalization,
                "channel_a": a,
                "channel_b": b,
                "channel_a_dim": int(x.shape[1]),
                "channel_b_dim": int(y.shape[1]),
                "measurement": "state_flat_ridge_r2",
                "value": r2,
                "std": r2_std,
                "n": int(x.shape[0]),
                "sanity_knn_k": int(sanity_knn_k),
                "svcca_components": int(svcca_components),
                "extended_metric_repeats": int(extended_metric_repeats),
                "ridge_alpha": float(ridge_alpha),
                "ridge_splits": int(ridge_splits),
                "permutation_repeats": int(permutation_repeats),
                **r2_calibration,
            }
        )
        rows.append(
            {
                "representation": representation,
                "representation_components": int(representation_components),
                **representation_flags(representation),
                "control": control,
                "control_feature_source": control_feature_source,
                "normalization": normalization,
                "channel_a": b,
                "channel_b": a,
                "channel_a_dim": int(y.shape[1]),
                "channel_b_dim": int(x.shape[1]),
                "measurement": "state_flat_ridge_r2_reverse",
                "value": r2_reverse,
                "std": r2_reverse_std,
                "n": int(y.shape[0]),
                "sanity_knn_k": int(sanity_knn_k),
                "svcca_components": int(svcca_components),
                "extended_metric_repeats": int(extended_metric_repeats),
                "ridge_alpha": float(ridge_alpha),
                "ridge_splits": int(ridge_splits),
                "permutation_repeats": int(permutation_repeats),
                **r2_reverse_calibration,
            }
        )
        if normed_tensors is None:
            continue
        action_values = []
        action_ns = []
        rng = np.random.default_rng(int(seed) + 303)
        null_by_repeat: list[list[float]] = [[] for _ in range(int(permutation_repeats))]
        for local_pos in range(int(positions.shape[0])):
            xa = np.asarray(normed_tensors[a][:, local_pos, :], dtype=np.float32)
            yb = np.asarray(normed_tensors[b][:, local_pos, :], dtype=np.float32)
            sx = cosine_similarity_matrix(xa)
            sy = cosine_similarity_matrix(yb)
            score, n = rsa_spearman_from_similarity(sx, sy)
            action_values.append(score)
            action_ns.append(n)
            for repeat_i in range(int(permutation_repeats)):
                perm = rng.permutation(yb.shape[0])
                null_by_repeat[repeat_i].append(rsa_spearman_from_similarity(sx, sy[np.ix_(perm, perm)])[0])
        clean = np.asarray(action_values, dtype=np.float64)
        clean = clean[np.isfinite(clean)]
        observed = float(clean.mean()) if clean.size else float("nan")
        action_null_values = []
        for values in null_by_repeat:
            null_clean = np.asarray(values, dtype=np.float64)
            null_clean = null_clean[np.isfinite(null_clean)]
            action_null_values.append(float(null_clean.mean()) if null_clean.size else float("nan"))
        rows.append(
            {
                "representation": representation,
                "representation_components": int(representation_components),
                **representation_flags(representation),
                "control": control,
                "control_feature_source": control_feature_source,
                "normalization": normalization,
                "channel_a": a,
                "channel_b": b,
                "channel_a_dim": int(normed_tensors[a].shape[-1]),
                "channel_b_dim": int(normed_tensors[b].shape[-1]),
                "measurement": "action_conditioned_rsa_spearman_mean",
                "value": observed,
                "std": float(clean.std(ddof=0)) if clean.size else float("nan"),
                "n": int(len(action_values)),
                "sanity_knn_k": int(sanity_knn_k),
                "svcca_components": int(svcca_components),
                "extended_metric_repeats": int(extended_metric_repeats),
                "ridge_alpha": float(ridge_alpha),
                "ridge_splits": int(ridge_splits),
                "permutation_repeats": int(permutation_repeats),
                **calibration_fields(observed, action_null_values),
            }
        )
    return rows


def literature_metric_rows_for_features(
    *,
    representation: str,
    representation_components: int,
    control: str,
    normalization: str,
    features: Mapping[str, np.ndarray],
    pairs: Sequence[tuple[str, str]],
    sanity_knn_k: int,
    svcca_components: int,
    repeats: int,
    seed: int,
    control_feature_source: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for a, b in pairs:
        if a not in features or b not in features:
            continue
        x = np.asarray(features[a], dtype=np.float32)
        y = np.asarray(features[b], dtype=np.float32)
        cycle_value, cycle_calibration = calibrated_cycle_knn(
            x,
            y,
            k=int(sanity_knn_k),
            repeats=int(repeats),
            seed=int(seed) + 251,
        )
        cknna_value, cknna_calibration = calibrated_cknna(
            x,
            y,
            k=int(sanity_knn_k),
            repeats=int(repeats),
            seed=int(seed) + 252,
        )
        svcca_value, svcca_n, svcca_calibration = calibrated_svcca_mean(
            x,
            y,
            components=int(svcca_components),
            repeats=int(repeats),
            seed=int(seed) + 253,
        )
        full_cca_components = max(1, min(int(x.shape[0] - 1), int(x.shape[1]), int(y.shape[1])))
        cca_value, cca_n, cca_calibration = calibrated_svcca_mean(
            x,
            y,
            components=full_cca_components,
            repeats=int(repeats),
            seed=int(seed) + 254,
        )
        for measurement, value, n_value, components, calibration in [
            ("cycle_knn_overlap", cycle_value, int(x.shape[0]), int(sanity_knn_k), cycle_calibration),
            ("cknna_linear_cka", cknna_value, int(x.shape[0]), int(sanity_knn_k), cknna_calibration),
            ("svcca_mean_corr", svcca_value, int(svcca_n), int(svcca_components), svcca_calibration),
            ("cca_mean_corr", cca_value, int(cca_n), int(full_cca_components), cca_calibration),
        ]:
            rows.append(
                {
                    "representation": representation,
                    "representation_components": int(representation_components),
                    **representation_flags(representation),
                    "control": control,
                    "control_feature_source": control_feature_source,
                    "normalization": normalization,
                    "channel_a": a,
                    "channel_b": b,
                    "channel_a_dim": int(x.shape[1]),
                    "channel_b_dim": int(y.shape[1]),
                    "measurement": measurement,
                    "value": value,
                    "std": float("nan"),
                    "n": int(n_value),
                    "metric_parameter": int(components),
                    "sanity_knn_k": int(sanity_knn_k),
                    "svcca_components": int(svcca_components),
                    "permutation_repeats": int(repeats),
                    "diagnostic_only": True,
                    "primary_gate_eligible": False,
                    **calibration,
                }
            )
    return rows


def decision_summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    key_rows = [
        r
        for r in rows
        if str(r["control"]) == "action_effect_heldout_signature"
        and str(r["channel_a"]) == "rgb"
        and str(r["channel_b"]) == "range"
    ]
    out = []
    for rep in sorted({str(r["representation"]) for r in key_rows}):
        for norm in sorted({str(r["normalization"]) for r in key_rows if str(r["representation"]) == rep}):
            subset = [
                r
                for r in key_rows
                if str(r["representation"]) == rep and str(r["normalization"]) == norm
            ]
            values = np.asarray([float(r["chance_adjusted_overlap"]) for r in subset], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            out.append(
                {
                    "representation": rep,
                    "normalization": norm,
                    "n_conditions": int(values.size),
                    "rgb_range_adjusted_mean": float(values.mean()),
                    "rgb_range_adjusted_min": float(values.min()),
                    "rgb_range_adjusted_max": float(values.max()),
                    "rgb_range_positive_fraction": float(np.mean(values > 0.0)),
                    "stable_positive": bool(values.min() > 0.0),
                }
            )
    return out


def main() -> None:
    args = parse_args()
    data = load_npz(args.data)
    channels = [str(x) for x in args.channels]
    leakage = sorted(set(channels) & diagnostic_only_channels(data))
    if leakage:
        raise ValueError(f"Stage B.6 excludes diagnostic-only channels: {leakage}")

    sample_indices = validate_dense_stage0_data(data, channels, require_detect=True)
    static_sample_indices = validate_dense_static_data(data, channels)
    state_ids, action_ids, dense_rows, full_rows = complete_state_action_matrix(
        data,
        sample_indices,
        max_states=int(args.max_states),
        seed=int(args.seed),
    )
    probe_set = {int(x) for x in args.probe_action_ids}
    probe_pos = np.asarray([i for i, action_id in enumerate(action_ids.tolist()) if int(action_id) in probe_set], dtype=np.int64)
    heldout_pos = np.asarray([i for i, action_id in enumerate(action_ids.tolist()) if int(action_id) not in probe_set], dtype=np.int64)
    if probe_pos.size == 0 or heldout_pos.size == 0:
        raise ValueError("Probe split must select at least one action and leave held-out actions.")
    full_action_types = np.asarray(data["action_type"]).astype(str)[full_rows[0]]
    requested_norms = [str(x) for x in args.normalization_modes]
    missing_probe_types = sorted(set(full_action_types[heldout_pos].tolist()) - set(full_action_types[probe_pos].tolist()))
    if "probe_action_type_apply" in requested_norms and missing_probe_types:
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

    representations = [str(x) for x in args.representations]
    representation_tensors: dict[str, dict[str, np.ndarray]] = {}
    representation_components: dict[str, int] = {}
    if "pca_probe_only" in representations:
        representation_tensors["pca_probe_only"] = embedding_tensors_from_model(
            data,
            encoders,
            channels,
            prefix="delta",
            dense_rows=dense_rows,
        )
        representation_components["pca_probe_only"] = pca_component_count(encoders)
    raw = raw_tensors(data, channels, prefix="delta", dense_rows=dense_rows)
    if "raw_delta" in representations:
        representation_tensors["raw_delta"] = raw
        representation_components["raw_delta"] = int(raw[channels[0]].shape[-1])
    if "random_projection" in representations:
        representation_tensors["random_projection"] = random_project_tensors(
            raw,
            dim=int(args.random_projection_dim),
            seed=int(args.random_projection_seed),
        )
        representation_components["random_projection"] = int(args.random_projection_dim)
    if "pca_all_action" in representations:
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
        validate_all_action_model(all_metadata, [int(x) for x in action_ids.tolist()])
        representation_tensors["pca_all_action"] = embedding_tensors_from_model(
            data,
            all_encoders,
            channels,
            prefix="delta",
            dense_rows=dense_rows,
        )
        representation_components["pca_all_action"] = pca_component_count(all_encoders)

    static_dense_rows = dense_rows_for_sample_ids(static_sample_indices, full_rows)
    static_tensors = embedding_tensors_from_model(
        data,
        static_encoders,
        channels,
        prefix="obs0",
        dense_rows=static_dense_rows,
    )
    _, fraction_rows, thresholds = state_pair_strata(
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
    pairs = list(DEFAULT_REDUNDANCY_PAIRS) + list(TARGET_PAIRS) + parse_pairs(args.target_pairs)
    pairs = list(dict.fromkeys(pairs))
    sanity_pairs = parse_pairs(args.target_pairs)

    sensitivity_rows: list[dict[str, object]] = []
    corr_rows: list[dict[str, object]] = []
    tie_rows: list[dict[str, object]] = []
    sanity_rows: list[dict[str, object]] = []
    literature_metric_rows: list[dict[str, object]] = []
    diff_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(int(args.bootstrap_seed))

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
        static_neighbors_by_k = {
            int(k): neighbor_sets(static_features, k=int(k), batch_size=int(args.batch_size))
            for k in args.k_values
        }
        for rep, tensors in representation_tensors.items():
            features = features_from_tensors(
                tensors,
                channels,
                heldout_pos,
                mode=norm,
                action_types=full_action_types,
                reference_positions=ref_pos,
            )
            shuffled = normalized_then_shuffled_features_from_tensors(
                tensors,
                channels,
                heldout_pos,
                mode=norm,
                action_types=full_action_types,
                reference_positions=ref_pos,
                seed=int(args.seed) + 7001,
            )
            sanity_rows.extend(
                sanity_rows_for_features(
                    representation=rep,
                    representation_components=representation_components.get(rep, 0),
                    control="action_effect_heldout_signature",
                    normalization=norm,
                    features=features,
                    tensors=tensors,
                    positions=heldout_pos,
                    action_types=full_action_types,
                    reference_positions=ref_pos,
                    pairs=sanity_pairs,
                    ridge_alpha=float(args.ridge_alpha),
                    ridge_splits=int(args.ridge_splits),
                    sanity_knn_k=int(args.sanity_knn_k),
                    svcca_components=int(args.svcca_components),
                    extended_metric_repeats=int(args.extended_metric_repeats),
                    permutation_repeats=int(args.permutation_repeats),
                    seed=int(args.permutation_seed) + int(args.seed) * 1009 + 9001,
                    control_feature_source="state_flat_action_effect",
                )
            )
            literature_metric_rows.extend(
                literature_metric_rows_for_features(
                    representation=rep,
                    representation_components=representation_components.get(rep, 0),
                    control="action_effect_heldout_signature",
                    normalization=norm,
                    features=features,
                    pairs=sanity_pairs,
                    sanity_knn_k=int(args.sanity_knn_k),
                    svcca_components=int(args.svcca_components),
                    repeats=int(args.extended_metric_repeats),
                    seed=int(args.permutation_seed) + int(args.seed) * 1009 + 9101,
                    control_feature_source="state_flat_action_effect",
                )
            )
            sanity_rows.extend(
                sanity_rows_for_features(
                    representation=rep,
                    representation_components=representation_components.get(rep, 0),
                    control="static_heldout_signature",
                    normalization=norm,
                    features=static_features,
                    tensors=None,
                    positions=heldout_pos,
                    action_types=full_action_types,
                    reference_positions=ref_pos,
                    pairs=sanity_pairs,
                    ridge_alpha=float(args.ridge_alpha),
                    ridge_splits=int(args.ridge_splits),
                    sanity_knn_k=int(args.sanity_knn_k),
                    svcca_components=int(args.svcca_components),
                    extended_metric_repeats=int(args.extended_metric_repeats),
                    permutation_repeats=int(args.permutation_repeats),
                    seed=int(args.permutation_seed) + int(args.seed) * 1009 + 9002,
                    control_feature_source="state_flat_static",
                )
            )
            literature_metric_rows.extend(
                literature_metric_rows_for_features(
                    representation=rep,
                    representation_components=representation_components.get(rep, 0),
                    control="static_heldout_signature",
                    normalization=norm,
                    features=static_features,
                    pairs=sanity_pairs,
                    sanity_knn_k=int(args.sanity_knn_k),
                    svcca_components=int(args.svcca_components),
                    repeats=int(args.extended_metric_repeats),
                    seed=int(args.permutation_seed) + int(args.seed) * 1009 + 9102,
                    control_feature_source="state_flat_static",
                )
            )
            sanity_rows.extend(
                sanity_rows_for_features(
                    representation=rep,
                    representation_components=representation_components.get(rep, 0),
                    control="action_column_shuffled_heldout",
                    normalization=norm,
                    features=shuffled,
                    tensors=None,
                    positions=heldout_pos,
                    action_types=full_action_types,
                    reference_positions=ref_pos,
                    pairs=sanity_pairs,
                    ridge_alpha=float(args.ridge_alpha),
                    ridge_splits=int(args.ridge_splits),
                    sanity_knn_k=int(args.sanity_knn_k),
                    svcca_components=int(args.svcca_components),
                    extended_metric_repeats=int(args.extended_metric_repeats),
                    permutation_repeats=int(args.permutation_repeats),
                    seed=int(args.permutation_seed) + int(args.seed) * 1009 + 9003,
                    control_feature_source="state_flat_normalized_then_action_shuffled",
                )
            )
            literature_metric_rows.extend(
                literature_metric_rows_for_features(
                    representation=rep,
                    representation_components=representation_components.get(rep, 0),
                    control="action_column_shuffled_heldout",
                    normalization=norm,
                    features=shuffled,
                    pairs=sanity_pairs,
                    sanity_knn_k=int(args.sanity_knn_k),
                    svcca_components=int(args.svcca_components),
                    repeats=int(args.extended_metric_repeats),
                    seed=int(args.permutation_seed) + int(args.seed) * 1009 + 9103,
                    control_feature_source="state_flat_normalized_then_action_shuffled",
                )
            )

            for k in [int(x) for x in args.k_values]:
                static_neighbors = static_neighbors_by_k[k]
                shuffled_neighbors = neighbor_sets(shuffled, k=k, batch_size=int(args.batch_size))
                for eps in [float(x) for x in args.jitter_epsilons]:
                    seeds = [-1] if eps == 0.0 else [int(x) for x in args.jitter_seeds]
                    for jitter_seed in seeds:
                        current = jittered_features(
                            features,
                            epsilon=eps,
                            seed=(int(args.seed) * 1000003 + int(jitter_seed)),
                        )
                        current_neighbors = neighbor_sets(current, k=k, batch_size=int(args.batch_size))
                        for row in feature_diagnostic_rows(current, feature_set="heldout", normalization=norm, k=k):
                            tie_rows.append(
                                {
                                    **row,
                                    "representation": rep,
                                    "representation_components": representation_components.get(rep, 0),
                                    "jitter_epsilon": eps,
                                    "jitter_seed": int(jitter_seed),
                                }
                            )
                        per_query_by_pair = {}
                        for control, neighbors in [
                            ("action_effect_heldout_signature", current_neighbors),
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
                                    neighbors=neighbors,
                                    k=k,
                                )
                                sensitivity_rows.append(
                                    {
                                        **base,
                                        "representation_components": representation_components.get(rep, 0),
                                        "jitter_epsilon": eps,
                                        "jitter_seed": int(jitter_seed),
                                    }
                                )
                                if control == "action_effect_heldout_signature":
                                    per_query_by_pair[(a, b)] = scores
                        for row in [
                            row
                            for row in corr_rows_for_scores_local(
                                representation=rep,
                                representation_components=representation_components.get(rep, 0),
                                control="action_effect_heldout_signature",
                                normalization=norm,
                                pair_scores=score_by_pair,
                                per_query_by_pair=per_query_by_pair,
                                k=k,
                                jitter_epsilon=eps,
                                jitter_seed=int(jitter_seed),
                            )
                        ]:
                            corr_rows.append(row)
                        for a, b in pairs:
                            if a not in channels or b not in channels:
                                continue
                            ae = per_query_scores(current_neighbors[a], current_neighbors[b], k=k)
                            st = per_query_scores(static_neighbors[a], static_neighbors[b], k=k)
                            sh = per_query_scores(shuffled_neighbors[a], shuffled_neighbors[b], k=k)
                            for comparison, control_scores in [
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
                                        "representation_components": representation_components.get(rep, 0),
                                        **representation_flags(rep),
                                        "comparison": comparison,
                                        "channel_a": a,
                                        "channel_b": b,
                                        "normalization": norm,
                                        "normalization_primary": bool(normalization_is_primary(norm)),
                                        "normalization_transductive": bool(normalization_is_transductive(norm)),
                                        "k": int(k),
                                        "jitter_epsilon": eps,
                                        "jitter_seed": int(jitter_seed),
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

    out = ensure_dir(args.out)
    reports = ensure_dir(out / "reports")
    write_csv(reports / "b6_knn_sensitivity.csv", sensitivity_rows)
    write_csv(reports / "b6_paired_differences.csv", diff_rows)
    write_csv(reports / "b6_observability_score_correlation_by_k_jitter.csv", corr_rows)
    write_csv(reports / "b6_feature_tie_by_k_jitter.csv", tie_rows)
    write_csv(reports / "b6_measurement_sanity.csv", sanity_rows)
    write_csv(reports / "b6_literature_metrics.csv", literature_metric_rows)
    write_csv(reports / "b6_decision_summary.csv", decision_summary_rows(sensitivity_rows))
    save_json(
        {
            "data": str(Path(args.data)),
            "model": str(Path(args.model)),
            "static_model": str(Path(args.static_model)),
            "all_action_model": str(Path(args.all_action_model)) if args.all_action_model else None,
            "channels": channels,
            "representations": sorted(representation_tensors),
            "normalization_modes": [str(x) for x in args.normalization_modes],
            "k_values": [int(x) for x in args.k_values],
            "jitter_epsilons": [float(x) for x in args.jitter_epsilons],
            "jitter_seeds": [int(x) for x in args.jitter_seeds],
            "permutation_repeats": int(args.permutation_repeats),
            "extended_metric_repeats": int(args.extended_metric_repeats),
            "sanity_knn_k": int(args.sanity_knn_k),
            "svcca_components": int(args.svcca_components),
            "n_states": int(state_ids.shape[0]),
            "n_probe_actions": int(probe_pos.size),
            "n_heldout_actions": int(heldout_pos.size),
            "strata_thresholds": thresholds,
            "notes": [
                "B.6 is diagnostic: it checks stability of the B.5 rgb-range signal.",
                "Stage C remains blocked unless k, jitter, PCA/component, and control diagnostics are stable.",
            ],
        },
        reports / "stageb6_summary.json",
    )
    print(f"Wrote Stage B.6 diagnostics to {out}")


def corr_rows_for_scores_local(
    *,
    representation: str,
    representation_components: int,
    control: str,
    normalization: str,
    pair_scores: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    per_query_by_pair: Mapping[tuple[str, str], np.ndarray],
    k: int,
    jitter_epsilon: float,
    jitter_seed: int,
) -> list[dict[str, object]]:
    rows = []
    for pair, query_scores in per_query_by_pair.items():
        for score_name, score_values in pair_scores[pair].items():
            pearson, spearman, n = corr(score_values, query_scores)
            rows.append(
                {
                    "representation": representation,
                    "representation_components": int(representation_components),
                    "control": control,
                    "normalization": normalization,
                    "normalization_primary": bool(normalization_is_primary(normalization)),
                    "normalization_transductive": bool(normalization_is_transductive(normalization)),
                    "channel_a": pair[0],
                    "channel_b": pair[1],
                    "score": score_name,
                    "score_primary": bool(score_name == "detect_geom_rank_mean"),
                    "k": int(k),
                    "jitter_epsilon": float(jitter_epsilon),
                    "jitter_seed": int(jitter_seed),
                    "n": n,
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )
    return rows


if __name__ == "__main__":
    main()
