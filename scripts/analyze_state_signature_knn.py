#!/usr/bin/env python
"""Stage B.2 state-level action-effect signature kNN alignment."""

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
    neighbor_overlap_stats,
    pairwise_overlap_matrix,
    pairwise_stratified_overlap_rows,
)
from sae_align.analysis.strata import (
    channel_blind_masks,
    diagnostic_only_channels,
    stage0_dataset_fingerprint,
    validate_dense_stage0_data,
    validate_dense_static_data,
)
from sae_align.models import load_transition_encoders
from sae_align.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model", type=str, required=True, help="Action-effect encoder trained on delta_<channel>.")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--static-model", type=str, default=None, help="Optional static encoder trained on obs0_<channel>.")
    p.add_argument("--channels", nargs="*", default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-states", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe-fraction", type=float, default=0.5)
    p.add_argument("--probe-action-ids", nargs="*", type=int, default=None)
    p.add_argument("--detectability-quantile", type=float, default=0.10)
    p.add_argument("--regular-state-threshold", type=float, default=0.60)
    p.add_argument("--blind-state-threshold", type=float, default=0.60)
    p.add_argument("--physical-state-threshold", type=float, default=0.60)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--bootstrap-repeats", type=int, default=0)
    p.add_argument("--bootstrap-seed", type=int, default=0)
    p.add_argument("--no-per-action-normalize", action="store_true")
    p.add_argument(
        "--allow-leakage-diagnostic",
        action="store_true",
        help="Allow diagnostic-only post-action channels. Do not use such rows as primary Stage B.2 evidence.",
    )
    p.add_argument(
        "--allow-cross-data-model",
        action="store_true",
        help="Allow encoder metadata to point at a different Stage 0 data path.",
    )
    p.add_argument(
        "--allow-all-action-trained-model",
        action="store_true",
        help="Allow action-effect encoders not explicitly fitted on the probe action IDs. This weakens held-out action claims.",
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


def complete_state_action_matrix(
    data: Mapping[str, np.ndarray],
    sample_indices: np.ndarray,
    *,
    state_ids: np.ndarray | None = None,
    action_ids: np.ndarray | None = None,
    max_states: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return dense row positions and full sample IDs for complete state/action blocks."""
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    full_state = np.asarray(data["state_id"], dtype=np.int64)[sample_indices]
    full_action = np.asarray(data["action_id"], dtype=np.int64)[sample_indices]
    required_actions = (
        np.asarray(action_ids, dtype=np.int64)
        if action_ids is not None
        else np.asarray(sorted(np.unique(np.asarray(data["action_id"], dtype=np.int64)).tolist()), dtype=np.int64)
    )
    candidate_states = (
        np.asarray(state_ids, dtype=np.int64)
        if state_ids is not None
        else np.asarray(sorted(np.unique(full_state).tolist()), dtype=np.int64)
    )
    pairs = [(int(s), int(a)) for s, a in zip(full_state.tolist(), full_action.tolist())]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Dense Stage 0 data contains duplicate (state_id, action_id) rows.")
    row_for = {pair: i for i, pair in enumerate(pairs)}
    complete_states = []
    for state in candidate_states.tolist():
        if all((int(state), int(action)) in row_for for action in required_actions.tolist()):
            complete_states.append(int(state))
    if not complete_states:
        raise ValueError(
            "No complete state x action-bank blocks found in dense deltas. "
            "Generate Stage 0 data with --dense-sampling full-states."
        )
    states = np.asarray(complete_states, dtype=np.int64)
    if max_states is not None and states.shape[0] > int(max_states):
        rng = np.random.default_rng(seed)
        states = np.sort(rng.choice(states, size=int(max_states), replace=False))
    dense_rows = np.asarray(
        [[row_for[(int(state), int(action))] for action in required_actions.tolist()] for state in states.tolist()],
        dtype=np.int64,
    )
    full_rows = sample_indices[dense_rows]
    return states, required_actions, dense_rows, full_rows


def dense_rows_for_sample_ids(sample_indices: np.ndarray, full_rows: np.ndarray) -> np.ndarray:
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    row_for = {int(sample_id): i for i, sample_id in enumerate(sample_indices.tolist())}
    rows = []
    for sample_id in np.asarray(full_rows, dtype=np.int64).reshape(-1).tolist():
        if int(sample_id) not in row_for:
            raise ValueError(f"Static dense data is missing sample id {int(sample_id)}.")
        rows.append(row_for[int(sample_id)])
    return np.asarray(rows, dtype=np.int64).reshape(np.asarray(full_rows).shape)


def split_action_positions(n_actions: int, probe_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n_actions < 2:
        raise ValueError("Stage B.2 requires at least two actions for probe/test splitting.")
    n_probe = int(round(float(probe_fraction) * n_actions))
    n_probe = max(1, min(n_actions - 1, n_probe))
    rng = np.random.default_rng(seed)
    probe = np.sort(rng.choice(np.arange(n_actions), size=n_probe, replace=False))
    test = np.asarray([i for i in range(n_actions) if i not in set(probe.tolist())], dtype=np.int64)
    return probe.astype(np.int64), test.astype(np.int64)


def split_action_positions_from_ids(
    action_ids: np.ndarray,
    probe_action_ids: Sequence[int] | None,
    probe_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if probe_action_ids is None:
        return split_action_positions(int(action_ids.shape[0]), probe_fraction, seed)
    action_ids = np.asarray(action_ids, dtype=np.int64)
    wanted = np.asarray(probe_action_ids, dtype=np.int64)
    missing = sorted(set(wanted.tolist()) - set(action_ids.tolist()))
    if missing:
        raise ValueError(f"Probe action IDs are not present in the dense action bank: {missing}")
    probe = np.asarray([i for i, action_id in enumerate(action_ids.tolist()) if int(action_id) in set(wanted.tolist())])
    if probe.shape[0] == 0 or probe.shape[0] >= action_ids.shape[0]:
        raise ValueError("--probe-action-ids must select at least one and leave at least one held-out action.")
    test = np.asarray([i for i in range(action_ids.shape[0]) if i not in set(probe.tolist())], dtype=np.int64)
    return np.sort(probe).astype(np.int64), test.astype(np.int64)


def validate_model_data_path(metadata: Mapping[str, object], data_path: str, *, allow_cross_data_model: bool) -> None:
    if allow_cross_data_model:
        return
    model_data = metadata.get("data")
    if not model_data:
        raise ValueError("Encoder metadata has no data path. Use --allow-cross-data-model only for diagnostics.")
    if Path(str(model_data)).resolve() != Path(data_path).resolve():
        raise ValueError(
            "Encoder metadata data path does not match --data. "
            "Use --allow-cross-data-model only for explicitly labeled diagnostics."
        )


def validate_model_fingerprint(
    metadata: Mapping[str, object],
    data: Mapping[str, np.ndarray],
    channels: Sequence[str],
    sample_indices: np.ndarray,
    *,
    prefix: str,
    allow_cross_data_model: bool,
) -> None:
    if allow_cross_data_model:
        return
    expected = metadata.get("stage0_fingerprint")
    if not expected:
        raise ValueError("Encoder metadata has no Stage 0 fingerprint. Re-train the encoder or use diagnostics only.")
    actual = stage0_dataset_fingerprint(data, channels, sample_indices, prefix=prefix)
    if str(expected) != str(actual):
        raise ValueError(
            "Encoder Stage 0 fingerprint does not match --data. "
            "Use --allow-cross-data-model only for explicitly labeled diagnostics."
        )


def validate_probe_trained_model(
    metadata: Mapping[str, object],
    probe_action_ids: Sequence[int],
    *,
    allow_all_action_trained_model: bool,
) -> str:
    train_action_ids = metadata.get("actual_train_action_ids", metadata.get("train_action_ids"))
    train_counts = metadata.get("actual_train_action_counts")
    if train_action_ids is None:
        if allow_all_action_trained_model:
            return "all_or_unspecified"
        raise ValueError(
            "Stage B.2 primary held-out action evidence requires encoders fitted only on probe actions. "
            "Re-train with --train-action-ids or pass --allow-all-action-trained-model for diagnostics."
        )
    train_set = {int(x) for x in train_action_ids}
    probe_set = {int(x) for x in probe_action_ids}
    if train_set != probe_set:
        if allow_all_action_trained_model:
            return "mismatch_allowed"
        raise ValueError(
            f"Encoder train_action_ids {sorted(train_set)} do not match probe_action_ids {sorted(probe_set)}."
        )
    if isinstance(train_counts, dict):
        missing_counts = [action_id for action_id in probe_set if int(train_counts.get(str(action_id), 0)) <= 0]
        if missing_counts:
            raise ValueError(f"Encoder has no fitted rows for probe action IDs: {sorted(missing_counts)}")
    return "probe_only"


def normalize_signature_tensor(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return ((x - mean) / (std + 1e-6)).astype(np.float32)


def signature_features(
    data: Mapping[str, np.ndarray],
    encoders: Mapping[str, object],
    channels: Sequence[str],
    *,
    prefix: str,
    dense_rows: np.ndarray,
    action_positions: np.ndarray,
    normalize_per_action: bool,
    shuffle_columns: bool,
    seed: int,
) -> Dict[str, np.ndarray]:
    features: Dict[str, np.ndarray] = {}
    rng = np.random.default_rng(seed)
    n_states, n_actions = dense_rows.shape
    flat_rows = dense_rows.reshape(-1)
    for channel in channels:
        if f"{prefix}_{channel}" not in data:
            raise KeyError(f"Missing {prefix}_{channel} for state-level signature analysis.")
        emb = encoders[channel].transform(data[f"{prefix}_{channel}"][flat_rows])
        tensor = emb.reshape(n_states, n_actions, -1)[:, action_positions, :].copy()
        if shuffle_columns:
            for i in range(tensor.shape[0]):
                tensor[i] = tensor[i, rng.permutation(tensor.shape[1])]
        if normalize_per_action:
            tensor = normalize_signature_tensor(tensor)
        features[channel] = tensor.reshape(tensor.shape[0], -1).astype(np.float32)
    return features


def state_pair_strata(
    data: Mapping[str, np.ndarray],
    channels: Sequence[str],
    full_rows: np.ndarray,
    action_positions: np.ndarray,
    *,
    detectability_quantile: float,
    regular_threshold: float,
    blind_threshold: float,
    physical_threshold: float,
) -> tuple[dict[tuple[str, str], dict[str, np.ndarray]], list[Dict[str, object]], dict[str, float]]:
    sample_indices = full_rows.reshape(-1)
    physical, blind, regular, thresholds = channel_blind_masks(
        data,
        channels,
        sample_indices,
        threshold_quantile=detectability_quantile,
        threshold_sample_indices=full_rows[:, action_positions].reshape(-1),
    )
    n_states, n_actions = full_rows.shape
    physical_m = physical.reshape(n_states, n_actions)[:, action_positions]
    blind_m = {ch: mask.reshape(n_states, n_actions)[:, action_positions] for ch, mask in blind.items()}
    regular_m = {ch: mask.reshape(n_states, n_actions)[:, action_positions] for ch, mask in regular.items()}
    pair_strata: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    fraction_rows: list[Dict[str, object]] = []
    for i, a in enumerate(channels):
        for b in channels[i + 1 :]:
            regular_both_fraction = np.mean(np.logical_and(regular_m[a], regular_m[b]), axis=1)
            blind_either_fraction = np.mean(np.logical_or(blind_m[a], blind_m[b]), axis=1)
            blind_both_fraction = np.mean(np.logical_and(blind_m[a], blind_m[b]), axis=1)
            physical_fraction = np.mean(physical_m, axis=1)
            regular_state = regular_both_fraction >= float(regular_threshold)
            blind_state = blind_either_fraction >= float(blind_threshold)
            pair_strata[(a, b)] = {
                "all": np.ones(n_states, dtype=bool),
                "regular_state": regular_state,
                "blind_state": blind_state,
                "mixed_state": np.logical_not(np.logical_or(regular_state, blind_state)),
                "physical_nonnull_state": physical_fraction >= float(physical_threshold),
            }
            for state_row in range(n_states):
                fraction_rows.append(
                    {
                        "channel_a": a,
                        "channel_b": b,
                        "state_row": state_row,
                        "regular_both_fraction": float(regular_both_fraction[state_row]),
                        "blind_either_fraction": float(blind_either_fraction[state_row]),
                        "blind_both_fraction": float(blind_both_fraction[state_row]),
                        "physical_nonnull_fraction": float(physical_fraction[state_row]),
                    }
                )
    return pair_strata, fraction_rows, thresholds


def neighbor_sets(features: Mapping[str, np.ndarray], *, k: int, batch_size: int) -> Dict[str, np.ndarray]:
    return {ch: cosine_knn_indices(x, k=k, batch_size=batch_size) for ch, x in features.items()}


def cross_split_rows(
    probe_neighbors: Mapping[str, np.ndarray],
    test_neighbors: Mapping[str, np.ndarray],
    pair_strata: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    *,
    k: int,
    control: str,
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    channels = list(probe_neighbors.keys())
    n = next(iter(probe_neighbors.values())).shape[0] if probe_neighbors else 0
    kk = min(k, next(iter(probe_neighbors.values())).shape[1]) if probe_neighbors else 0
    chance = float(kk / max(1, n - 1)) if n > 1 else float("nan")
    for i, a in enumerate(channels):
        for b in channels[i + 1 :]:
            for stratum, mask in pair_strata[(a, b)].items():
                mask = np.asarray(mask, dtype=bool)
                scores = symmetric_query_overlap_scores(
                    probe_neighbors[a],
                    test_neighbors[b],
                    probe_neighbors[b],
                    test_neighbors[a],
                    k=k,
                    query_mask=mask,
                )
                overlap = float(np.mean(scores)) if scores.size else float("nan")
                rows.append(
                    {
                        "control": control,
                        "channel_a": a,
                        "channel_b": b,
                        "stratum": stratum,
                        "n_queries": int(mask.sum()),
                        "n_valid_queries": int(scores.size),
                        "mean_effective_k": float(kk) if scores.size else float("nan"),
                        "overlap": overlap,
                        "random_expected_overlap": chance,
                        "chance_adjusted_overlap": overlap - chance if np.isfinite(overlap) else float("nan"),
                    }
                )
    return rows


def query_overlap_scores(a: np.ndarray, b: np.ndarray, *, k: int, query_mask: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    b = np.asarray(b)
    kk = min(a.shape[1], b.shape[1], int(k))
    if kk <= 0:
        return np.asarray([], dtype=np.float32)
    scores = []
    for i in np.nonzero(np.asarray(query_mask, dtype=bool))[0]:
        a_row = [int(v) for v in a[i, :kk].tolist() if int(v) >= 0]
        b_row = [int(v) for v in b[i, :kk].tolist() if int(v) >= 0]
        denom = min(kk, len(a_row), len(b_row))
        if denom:
            scores.append(len(set(a_row) & set(b_row)) / float(denom))
    return np.asarray(scores, dtype=np.float32)


def symmetric_query_overlap_scores(
    a1: np.ndarray,
    b1: np.ndarray,
    a2: np.ndarray,
    b2: np.ndarray,
    *,
    k: int,
    query_mask: np.ndarray,
) -> np.ndarray:
    a1 = np.asarray(a1)
    b1 = np.asarray(b1)
    a2 = np.asarray(a2)
    b2 = np.asarray(b2)
    kk = min(a1.shape[1], b1.shape[1], a2.shape[1], b2.shape[1], int(k))
    if kk <= 0:
        return np.asarray([], dtype=np.float32)
    scores = []
    for i in np.nonzero(np.asarray(query_mask, dtype=bool))[0]:
        per_direction = []
        for left, right in [(a1, b1), (a2, b2)]:
            left_row = [int(v) for v in left[i, :kk].tolist() if int(v) >= 0]
            right_row = [int(v) for v in right[i, :kk].tolist() if int(v) >= 0]
            denom = min(kk, len(left_row), len(right_row))
            if denom:
                per_direction.append(len(set(left_row) & set(right_row)) / float(denom))
        if per_direction:
            scores.append(float(np.mean(per_direction)))
    return np.asarray(scores, dtype=np.float32)


def bootstrap_mean_ci(scores: np.ndarray, *, repeats: int, seed: int) -> tuple[float, float, float]:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(scores))
    if repeats <= 0 or scores.size == 1:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(scores, size=(int(repeats), scores.size), replace=True)
    means = samples.mean(axis=1)
    return mean, float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def bootstrap_rows(
    neighbor_sets_by_control: Mapping[str, Mapping[str, np.ndarray]],
    pair_strata: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    *,
    k: int,
    repeats: int,
    seed: int,
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    if repeats <= 0:
        return rows
    controls = list(neighbor_sets_by_control.keys())
    if not controls:
        return rows
    first_neighbors = next(iter(next(iter(neighbor_sets_by_control.values())).values()))
    n = int(first_neighbors.shape[0])
    kk = min(int(k), int(first_neighbors.shape[1]))
    chance = float(kk / max(1, n - 1)) if n > 1 else float("nan")
    rng = np.random.default_rng(seed)
    for control in controls:
        neighbors = neighbor_sets_by_control[control]
        channels = list(neighbors.keys())
        for i, a in enumerate(channels):
            for b in channels[i + 1 :]:
                for stratum, mask in pair_strata[(a, b)].items():
                    scores = query_overlap_scores(neighbors[a], neighbors[b], k=k, query_mask=mask)
                    mean, low, high = bootstrap_mean_ci(
                        scores,
                        repeats=int(repeats),
                        seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                    )
                    rows.append(
                        {
                            "control": control,
                            "channel_a": a,
                            "channel_b": b,
                            "stratum": stratum,
                            "n_valid_queries": int(scores.size),
                            "overlap_mean": mean,
                            "overlap_ci95_low": low,
                            "overlap_ci95_high": high,
                            "random_expected_overlap": chance,
                            "chance_adjusted_mean": mean - chance if np.isfinite(mean) else float("nan"),
                            "chance_adjusted_ci95_low": low - chance if np.isfinite(low) else float("nan"),
                            "chance_adjusted_ci95_high": high - chance if np.isfinite(high) else float("nan"),
                            "bootstrap_repeats": int(repeats),
                        }
                    )
    return rows


def bootstrap_cross_rows(
    probe_neighbors: Mapping[str, np.ndarray],
    test_neighbors: Mapping[str, np.ndarray],
    pair_strata: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    *,
    k: int,
    repeats: int,
    seed: int,
    control: str,
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    if repeats <= 0:
        return rows
    channels = list(probe_neighbors.keys())
    n = next(iter(probe_neighbors.values())).shape[0] if probe_neighbors else 0
    kk = min(int(k), next(iter(probe_neighbors.values())).shape[1]) if probe_neighbors else 0
    chance = float(kk / max(1, n - 1)) if n > 1 else float("nan")
    rng = np.random.default_rng(seed)
    for i, a in enumerate(channels):
        for b in channels[i + 1 :]:
            for stratum, mask in pair_strata[(a, b)].items():
                scores = symmetric_query_overlap_scores(
                    probe_neighbors[a],
                    test_neighbors[b],
                    probe_neighbors[b],
                    test_neighbors[a],
                    k=k,
                    query_mask=mask,
                )
                mean, low, high = bootstrap_mean_ci(
                    scores,
                    repeats=int(repeats),
                    seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                )
                rows.append(
                    {
                        "control": control,
                        "channel_a": a,
                        "channel_b": b,
                        "stratum": stratum,
                        "n_valid_queries": int(scores.size),
                        "overlap_mean": mean,
                        "overlap_ci95_low": low,
                        "overlap_ci95_high": high,
                        "random_expected_overlap": chance,
                        "chance_adjusted_mean": mean - chance if np.isfinite(mean) else float("nan"),
                        "chance_adjusted_ci95_low": low - chance if np.isfinite(low) else float("nan"),
                        "chance_adjusted_ci95_high": high - chance if np.isfinite(high) else float("nan"),
                        "bootstrap_repeats": int(repeats),
                    }
                )
    return rows


def summarize_fraction_rows(fraction_rows: Sequence[Mapping[str, object]]) -> list[Dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in fraction_rows:
        pair = (str(row["channel_a"]), str(row["channel_b"]))
        for metric in [
            "regular_both_fraction",
            "blind_either_fraction",
            "blind_both_fraction",
            "physical_nonnull_fraction",
        ]:
            grouped.setdefault((pair[0], pair[1], metric), []).append(float(row[metric]))
    summary_rows: list[Dict[str, object]] = []
    for (a, b, metric), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float32)
        summary_rows.append(
            {
                "channel_a": a,
                "channel_b": b,
                "metric": metric,
                "n_states": int(arr.size),
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "q05": float(np.quantile(arr, 0.05)),
                "q10": float(np.quantile(arr, 0.10)),
                "q25": float(np.quantile(arr, 0.25)),
                "median": float(np.quantile(arr, 0.50)),
                "q75": float(np.quantile(arr, 0.75)),
                "q90": float(np.quantile(arr, 0.90)),
                "q95": float(np.quantile(arr, 0.95)),
                "max": float(np.max(arr)),
            }
        )
    return summary_rows


def static_gain_rows(
    action_rows: list[Dict[str, object]],
    static_rows: list[Dict[str, object]],
) -> list[Dict[str, object]]:
    def key(row: Mapping[str, object]) -> tuple[object, object, object]:
        return (row["channel_a"], row["channel_b"], row["stratum"])

    action_by_key = {key(row): row for row in action_rows}
    static_by_key = {key(row): row for row in static_rows}
    rows: list[Dict[str, object]] = []
    for item in sorted(set(action_by_key) & set(static_by_key)):
        arow = action_by_key[item]
        srow = static_by_key[item]
        action_adj = float(arow["chance_adjusted_overlap"])
        static_adj = float(srow["chance_adjusted_overlap"])
        action_overlap = float(arow["overlap"])
        static_overlap = float(srow["overlap"])
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
    encoders, metadata = load_transition_encoders(args.model)
    validate_model_data_path(metadata, args.data, allow_cross_data_model=bool(args.allow_cross_data_model))
    if metadata.get("feature_prefix", "delta") != "delta":
        raise ValueError("Stage B.2 action-effect model must be trained on delta_<channel> features.")
    channels = [str(x) for x in (args.channels or list(encoders.keys()))]
    missing = [ch for ch in channels if ch not in encoders or f"delta_{ch}" not in data]
    if missing:
        raise KeyError(f"Missing action-effect model/data for channels: {missing}")
    leakage_channels = sorted(set(channels) & diagnostic_only_channels(data))
    if leakage_channels and not args.allow_leakage_diagnostic:
        raise ValueError(f"Diagnostic-only channels are excluded from primary Stage B.2: {leakage_channels}.")

    sample_indices = validate_dense_stage0_data(data, channels, require_detect=True)
    validate_model_fingerprint(
        metadata,
        data,
        channels,
        sample_indices,
        prefix="delta",
        allow_cross_data_model=bool(args.allow_cross_data_model),
    )
    state_ids, action_ids, dense_rows, full_rows = complete_state_action_matrix(
        data,
        sample_indices,
        max_states=int(args.max_states),
        seed=int(args.seed),
    )
    probe_pos, test_pos = split_action_positions_from_ids(
        action_ids,
        args.probe_action_ids,
        probe_fraction=float(args.probe_fraction),
        seed=int(args.seed) + 11,
    )
    probe_action_ids = [int(action_ids[i]) for i in probe_pos.tolist()]
    heldout_action_ids = [int(action_ids[i]) for i in test_pos.tolist()]
    encoder_action_split = validate_probe_trained_model(
        metadata,
        probe_action_ids,
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
    normalize = not bool(args.no_per_action_normalize)
    probe_features = signature_features(
        data,
        encoders,
        channels,
        prefix="delta",
        dense_rows=dense_rows,
        action_positions=probe_pos,
        normalize_per_action=normalize,
        shuffle_columns=False,
        seed=int(args.seed) + 101,
    )
    test_features = signature_features(
        data,
        encoders,
        channels,
        prefix="delta",
        dense_rows=dense_rows,
        action_positions=test_pos,
        normalize_per_action=normalize,
        shuffle_columns=False,
        seed=int(args.seed) + 102,
    )
    shuffled_features = signature_features(
        data,
        encoders,
        channels,
        prefix="delta",
        dense_rows=dense_rows,
        action_positions=probe_pos,
        normalize_per_action=normalize,
        shuffle_columns=True,
        seed=int(args.seed) + 103,
    )
    probe_neighbors = neighbor_sets(probe_features, k=int(args.k), batch_size=int(args.batch_size))
    test_neighbors = neighbor_sets(test_features, k=int(args.k), batch_size=int(args.batch_size))
    shuffled_neighbors = neighbor_sets(shuffled_features, k=int(args.k), batch_size=int(args.batch_size))

    probe_names, probe_overlap = pairwise_overlap_matrix(probe_neighbors, k=int(args.k))
    test_names, test_overlap = pairwise_overlap_matrix(test_neighbors, k=int(args.k))
    probe_rows = pairwise_stratified_overlap_rows(
        probe_neighbors,
        pair_strata,
        k=int(args.k),
        control="action_effect_probe_signature",
    )
    test_rows = pairwise_stratified_overlap_rows(
        test_neighbors,
        pair_strata,
        k=int(args.k),
        control="action_effect_heldout_signature",
    )
    shuffled_rows = pairwise_stratified_overlap_rows(
        shuffled_neighbors,
        pair_strata,
        k=int(args.k),
        control="action_column_shuffled",
    )
    cross_rows = cross_split_rows(
        probe_neighbors,
        test_neighbors,
        pair_strata,
        k=int(args.k),
        control="probe_to_heldout_cross",
    )
    all_rows = probe_rows + test_rows + cross_rows + shuffled_rows
    bootstrap_ci_rows = bootstrap_rows(
        {
            "action_effect_probe_signature": probe_neighbors,
            "action_effect_heldout_signature": test_neighbors,
            "action_column_shuffled": shuffled_neighbors,
        },
        pair_strata,
        k=int(args.k),
        repeats=int(args.bootstrap_repeats),
        seed=int(args.bootstrap_seed),
    )
    bootstrap_ci_rows.extend(
        bootstrap_cross_rows(
            probe_neighbors,
            test_neighbors,
            pair_strata,
            k=int(args.k),
            repeats=int(args.bootstrap_repeats),
            seed=int(args.bootstrap_seed) + 17,
            control="probe_to_heldout_cross",
        )
    )

    static_rows: list[Dict[str, object]] = []
    gain_rows: list[Dict[str, object]] = []
    diagnostic_probe_gain_rows: list[Dict[str, object]] = []
    if args.static_model:
        static_encoders, static_metadata = load_transition_encoders(args.static_model)
        validate_model_data_path(
            static_metadata,
            args.data,
            allow_cross_data_model=bool(args.allow_cross_data_model),
        )
        if static_metadata.get("feature_prefix") != "obs0":
            raise ValueError("Static model must be trained on obs0_<channel> features.")
        missing_static = [ch for ch in channels if ch not in static_encoders or f"obs0_{ch}" not in data]
        if missing_static:
            raise KeyError(f"Missing static model/data for channels: {missing_static}")
        static_sample_indices = validate_dense_static_data(data, channels)
        validate_model_fingerprint(
            static_metadata,
            data,
            channels,
            static_sample_indices,
            prefix="obs0",
            allow_cross_data_model=bool(args.allow_cross_data_model),
        )
        static_dense_rows = dense_rows_for_sample_ids(static_sample_indices, full_rows)
        validate_probe_trained_model(
            static_metadata,
            probe_action_ids,
            allow_all_action_trained_model=bool(args.allow_all_action_trained_model),
        )
        static_probe_features = signature_features(
            data,
            static_encoders,
            channels,
            prefix="obs0",
            dense_rows=static_dense_rows,
            action_positions=probe_pos,
            normalize_per_action=normalize,
            shuffle_columns=False,
            seed=int(args.seed) + 201,
        )
        static_test_features = signature_features(
            data,
            static_encoders,
            channels,
            prefix="obs0",
            dense_rows=static_dense_rows,
            action_positions=test_pos,
            normalize_per_action=normalize,
            shuffle_columns=False,
            seed=int(args.seed) + 202,
        )
        static_probe_neighbors = neighbor_sets(static_probe_features, k=int(args.k), batch_size=int(args.batch_size))
        static_test_neighbors = neighbor_sets(static_test_features, k=int(args.k), batch_size=int(args.batch_size))
        static_probe_rows = pairwise_stratified_overlap_rows(
            static_probe_neighbors,
            pair_strata,
            k=int(args.k),
            control="static_probe_signature",
        )
        static_rows = pairwise_stratified_overlap_rows(
            static_test_neighbors,
            pair_strata,
            k=int(args.k),
            control="static_heldout_signature",
        )
        bootstrap_ci_rows.extend(
            bootstrap_rows(
                {
                    "static_probe_signature": static_probe_neighbors,
                    "static_heldout_signature": static_test_neighbors,
                },
                pair_strata,
                k=int(args.k),
                repeats=int(args.bootstrap_repeats),
                seed=int(args.bootstrap_seed) + 31,
            )
        )
        gain_rows = static_gain_rows(test_rows, static_rows)
        diagnostic_probe_gain_rows = static_gain_rows(probe_rows, static_probe_rows)
        all_rows.extend(static_probe_rows + static_rows)

    out = ensure_dir(args.out)
    report_dir = ensure_dir(out / "reports")
    write_matrix_csv(report_dir / "state_signature_probe_overlap.csv", probe_names, probe_overlap)
    write_matrix_csv(report_dir / "state_signature_heldout_overlap.csv", test_names, test_overlap)
    write_long_csv(report_dir / "state_signature_knn.csv", all_rows)
    write_long_csv(report_dir / "primary_state_signature_knn.csv", test_rows + cross_rows)
    write_long_csv(report_dir / "diagnostic_probe_signature_knn.csv", probe_rows)
    write_long_csv(report_dir / "heldout_action_signature_knn.csv", test_rows + cross_rows)
    write_long_csv(report_dir / "state_strata_fractions.csv", fraction_rows)
    write_long_csv(report_dir / "state_strata_fraction_summary.csv", summarize_fraction_rows(fraction_rows))
    write_long_csv(report_dir / "state_signature_bootstrap_ci.csv", bootstrap_ci_rows)
    write_long_csv(report_dir / "state_delta_minus_static_gain.csv", gain_rows)
    write_long_csv(report_dir / "diagnostic_probe_delta_minus_static_gain.csv", diagnostic_probe_gain_rows)
    save_json(
        {
            "data": str(Path(args.data)),
            "model": str(Path(args.model)),
            "static_model": str(Path(args.static_model)) if args.static_model else None,
            "model_metadata": metadata,
            "channels": channels,
            "k": int(args.k),
            "n_states": int(state_ids.shape[0]),
            "n_actions": int(action_ids.shape[0]),
            "state_ids_min": int(state_ids.min()) if state_ids.size else None,
            "state_ids_max": int(state_ids.max()) if state_ids.size else None,
            "action_ids": [int(x) for x in action_ids.tolist()],
            "probe_action_ids": probe_action_ids,
            "heldout_action_ids": heldout_action_ids,
            "encoder_action_split": encoder_action_split,
            "probe_fraction": float(args.probe_fraction),
            "detectability_quantile": float(args.detectability_quantile),
            "regular_state_threshold": float(args.regular_state_threshold),
            "blind_state_threshold": float(args.blind_state_threshold),
            "physical_state_threshold": float(args.physical_state_threshold),
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "per_action_normalize": normalize,
            "strata_thresholds": thresholds,
            "primary_controls": ["action_effect_heldout_signature", "probe_to_heldout_cross"],
            "diagnostic_controls": [
                "action_effect_probe_signature",
                "action_column_shuffled",
                "static_probe_signature",
                "static_heldout_signature",
            ],
            "controls": {
                "action_effect_probe_signature": True,
                "action_effect_heldout_signature": True,
                "probe_to_heldout_cross": True,
                "action_column_shuffled": True,
                "static_probe_signature": bool(static_rows),
                "static_heldout_signature": bool(static_rows),
            },
        },
        report_dir / "stageb2_state_signature_summary.json",
    )
    print(f"Wrote Stage B.2 state signature reports to {out}")


if __name__ == "__main__":
    main()
