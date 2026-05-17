#!/usr/bin/env python
"""Analyze Stage 0 observation-channel adequacy.

Outputs:
- blind-locus overlap and complementarity matrices;
- redundancy-probe R^2 matrices;
- action-type-conditioned effect/blind-rate reports;
- K-bootstrap and threshold-sweep reports;
- stage plots and summary JSON.

This script is deliberately representation-independent: Stage 0 strata are
computed from oracle world-state deltas and observation-level detectability,
not learned representations. This avoids circularity before Stage B.
"""

from __future__ import annotations

# Allow running scripts from a fresh clone before `pip install -e .`.
import sys
from pathlib import Path as _Path
_REPO_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from sae_align.analysis.metrics import binary_jaccard, random_projection, ridge_r2
from sae_align.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--k-sweep", nargs="*", type=int, default=[16, 32, 64, 128])
    p.add_argument("--k-bootstrap", type=int, default=20)
    p.add_argument("--threshold-quantiles", nargs="*", type=float, default=[0.05, 0.10, 0.20])
    p.add_argument("--default-threshold", type=float, default=0.10)
    p.add_argument("--projection-dim", type=int, default=64)
    p.add_argument("--max-probe-samples", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def normalize_schema(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Normalize older Stage-0 datasets to the current channel names."""
    channels = [str(x) for x in data["channels"].tolist()]
    if "event" not in channels:
        return data
    normalized = dict(data)
    new_channels = []
    for ch in channels:
        out_ch = "event_response" if ch == "event" else ch
        if out_ch not in new_channels:
            new_channels.append(out_ch)
    for suffix in ["detect", "delta", "example"]:
        old_key = f"{suffix}_event"
        new_key = f"{suffix}_event_response"
        if old_key in normalized and new_key not in normalized:
            normalized[new_key] = normalized[old_key]
    normalized["channels"] = np.array(new_channels)
    if "schema_version" not in normalized:
        normalized["schema_version"] = np.array(["stage0_legacy_event_alias"])
    return normalized


def validate_dense_deltas(data: Dict[str, np.ndarray], channels: Sequence[str]) -> Dict[str, object]:
    n = int(data["world_delta"].shape[0])
    if "delta_sample_indices" in data:
        dense_idx = np.asarray(data["delta_sample_indices"], dtype=np.int64)
    else:
        dense_idx = np.arange(n, dtype=np.int64)
    if dense_idx.ndim != 1:
        raise ValueError("delta_sample_indices must be a 1D array.")
    if len(dense_idx) == 0:
        raise ValueError("No dense delta samples are available.")
    if int(np.min(dense_idx)) < 0 or int(np.max(dense_idx)) >= n:
        raise ValueError("delta_sample_indices contains out-of-range sample IDs.")
    if len(np.unique(dense_idx)) != len(dense_idx):
        raise ValueError("delta_sample_indices contains duplicates.")
    for ch in channels:
        key = f"delta_{ch}"
        if key not in data:
            raise KeyError(f"Missing dense delta array {key}.")
        if int(data[key].shape[0]) != len(dense_idx):
            raise ValueError(f"{key} has {data[key].shape[0]} rows but delta_sample_indices has {len(dense_idx)}.")

    action_types = np.asarray(data["action_type"]).astype(str)
    full_types, full_counts = np.unique(action_types, return_counts=True)
    dense_types, dense_counts = np.unique(action_types[dense_idx], return_counts=True)
    dense_type_counts = {str(k): int(v) for k, v in zip(dense_types, dense_counts)}
    full_type_counts = {str(k): int(v) for k, v in zip(full_types, full_counts)}
    physical_nonnull = data["world_delta"] >= max(float(np.quantile(data["world_delta"], 0.10)), 1e-8)
    return {
        "dense_delta_samples": int(len(dense_idx)),
        "dense_delta_fraction": float(len(dense_idx) / n),
        "dense_physical_nonnull_rate": float(np.mean(physical_nonnull[dense_idx])),
        "full_physical_nonnull_rate_q10": float(np.mean(physical_nonnull)),
        "dense_action_type_counts": dense_type_counts,
        "full_action_type_counts": full_type_counts,
    }


def write_matrix_csv(path: Path, names: Sequence[str], mat: np.ndarray) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + list(names))
        for name, row in zip(names, mat):
            writer.writerow([name] + [f"{x:.6f}" if np.isfinite(x) else "nan" for x in row])


def write_long_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ridge_predict(x: np.ndarray, y: np.ndarray, alpha: float = 1.0, seed: int = 0, max_train: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = rng.permutation(n)
    if n > max_train:
        idx = idx[:max_train]
        n = len(idx)
    split = max(2, int(0.8 * n))
    train, test = idx[:split], idx[split:]
    xtr, ytr = x[train], y[train]
    xte, yte = x[test], y[test]
    xtrb = np.concatenate([xtr, np.ones((xtr.shape[0], 1), dtype=xtr.dtype)], axis=1)
    xteb = np.concatenate([xte, np.ones((xte.shape[0], 1), dtype=xte.dtype)], axis=1)
    xtx = xtrb.T @ xtrb
    reg = alpha * np.eye(xtx.shape[0], dtype=np.float32)
    reg[-1, -1] = 0.0
    w = np.linalg.solve(xtx + reg, xtrb.T @ ytr)
    return xteb @ w, yte


def plot_matrix(path: Path, names: Sequence[str], mat: np.ndarray, title: str, vmin: float = 0.0, vmax: float = 1.0) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(names)), max(6, 0.7 * len(names))))
    im = ax.imshow(mat, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _mean_blur_chw_batch(x: np.ndarray) -> np.ndarray:
    pad = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    out = np.zeros_like(x, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += pad[:, :, dy : dy + x.shape[2], dx : dx + x.shape[3]]
    return out / 9.0


def control_consistency_rows(data: Dict[str, np.ndarray]) -> List[Dict[str, object]]:
    """Direct consistency checks for known RGB-derived controls.

    This complements the learned redundancy probe. Random projection can obscure
    deterministic relationships when source and target have different raw
    shapes, e.g. RGB -> grayscale.
    """
    if "delta_rgb" not in data:
        return []
    rgb = np.asarray(data["delta_rgb"], dtype=np.float32)
    candidates: List[Tuple[str, np.ndarray]] = []
    if "delta_noisy_rgb" in data:
        candidates.append(("noisy_rgb", rgb))
    if "delta_gray_rgb" in data:
        gray = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        candidates.append(("gray_rgb", gray[:, None, :, :].astype(np.float32)))
    if "delta_blur_rgb" in data:
        candidates.append(("blur_rgb", _mean_blur_chw_batch(rgb).astype(np.float32)))

    rows: List[Dict[str, object]] = []
    for target, pred in candidates:
        true = np.asarray(data[f"delta_{target}"], dtype=np.float32)
        pred2 = pred.reshape(pred.shape[0], -1)
        true2 = true.reshape(true.shape[0], -1)
        ss_res = float(np.sum((true2 - pred2) ** 2))
        ss_tot = float(np.sum((true2 - true2.mean(axis=0, keepdims=True)) ** 2) + 1e-8)
        rows.append(
            {
                "source": "rgb",
                "target": target,
                "direct_r2": float(1.0 - ss_res / ss_tot),
                "mean_abs_error": float(np.mean(np.abs(true2 - pred2))),
                "n_dense_samples": int(true2.shape[0]),
            }
        )
    return rows


def compute_blind_masks(data: Dict[str, np.ndarray], channels: Sequence[str], q: float) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    wx = data["world_delta"]
    eps_x = float(np.quantile(wx, q))
    eps_x = max(eps_x, 1e-8)
    physical_nonnull = wx >= eps_x
    masks: Dict[str, np.ndarray] = {}
    thresholds = {"world": eps_x}
    for ch in channels:
        det = data[f"detect_{ch}"]
        det_base = det[physical_nonnull]
        tau = float(np.quantile(det_base, q)) if len(det_base) else 1e-8
        tau = max(tau, 1e-8)
        thresholds[ch] = tau
        masks[ch] = np.logical_and(physical_nonnull, det <= tau)
    return physical_nonnull, masks, thresholds


def jaccard_matrix(masks: Dict[str, np.ndarray], channels: Sequence[str]) -> np.ndarray:
    n = len(channels)
    mat = np.zeros((n, n), dtype=np.float32)
    for i, a in enumerate(channels):
        for j, b in enumerate(channels):
            mat[i, j] = binary_jaccard(masks[a], masks[b])
    return mat


def plot_effect_histograms(path: Path, data: Dict[str, np.ndarray], channels: Sequence[str]) -> None:
    ensure_dir(path.parent)
    n = len(channels) + 1
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.0))
    axes = np.array(axes).reshape(-1)
    axes[0].hist(data["world_delta"], bins=50)
    axes[0].set_title("world_delta")
    for ax, ch in zip(axes[1:], channels):
        ax.hist(data[f"detect_{ch}"], bins=50)
        ax.set_title(f"detect_{ch}")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_examples(path: Path, data: Dict[str, np.ndarray], channels: Sequence[str]) -> None:
    ensure_dir(path.parent)
    available = [ch for ch in channels if f"example_{ch}" in data]
    if not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 4))
    if len(available) == 1:
        axes = [axes]
    for ax, ch in zip(axes, available):
        arr = data[f"example_{ch}"][0]
        if arr.ndim == 3 and arr.shape[0] == 3:
            img = np.moveaxis(arr, 0, -1)
            ax.imshow(np.clip(img, 0, 1))
        elif arr.ndim == 3:
            ax.imshow(arr[0], cmap="viridis")
        elif arr.ndim == 1:
            ax.bar(np.arange(len(arr)), arr)
        else:
            ax.imshow(arr, cmap="viridis")
        ax.set_title(ch)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _project_by_raw_shape(data: Dict[str, np.ndarray], channels: Sequence[str], sample: np.ndarray, proj_dim: int, seed: int) -> Dict[str, np.ndarray]:
    """Project channel deltas, sharing projection matrices by raw flattened shape.

    ``sample`` indexes the stored dense-delta subset, not necessarily the full
    state-action table. This supports memory-aware Stage 0 datasets that store
    detectability for all samples but dense deltas only for a subset.
    """
    shape_seed: Dict[int, int] = {}
    embedded: Dict[str, np.ndarray] = {}
    next_seed_offset = 0
    for ch in channels:
        delta_key = f"delta_{ch}"
        if delta_key not in data:
            raise KeyError(f"Missing {delta_key}; regenerate with --max-delta-samples > 0.")
        x = data[delta_key][sample]
        flat_dim = int(np.prod(x.shape[1:]))
        if flat_dim not in shape_seed:
            shape_seed[flat_dim] = seed + 1000 + 97 * next_seed_offset
            next_seed_offset += 1
        embedded[ch] = random_projection(x, out_dim=proj_dim, seed=shape_seed[flat_dim])
    return embedded


def compute_redundancy_matrix(data: Dict[str, np.ndarray], channels: Sequence[str], proj_dim: int, max_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if "delta_sample_indices" in data:
        n_available = int(data["delta_sample_indices"].shape[0])
    else:
        n_available = int(data["world_delta"].shape[0])
    if n_available <= 0:
        raise ValueError("No dense delta samples available for redundancy probe.")
    sample = rng.choice(n_available, size=min(max_samples, n_available), replace=False)
    embedded = _project_by_raw_shape(data, channels, sample, proj_dim, seed)
    mat = np.zeros((len(channels), len(channels)), dtype=np.float32)
    for i, src in enumerate(channels):
        for j, tgt in enumerate(channels):
            if src == tgt:
                mat[i, j] = 1.0
            else:
                mat[i, j] = ridge_r2(embedded[src], embedded[tgt], alpha=1.0, seed=seed + 17 * i + j, max_train=min(max_samples, n_available))
    return mat

def k_sweep_bootstrap_rows(data: Dict[str, np.ndarray], channels: Sequence[str], ks: Sequence[int], q: float, seed: int, n_boot: int) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, float]]]:
    rng = np.random.default_rng(seed)
    action_ids = np.unique(data["action_id"])
    _, full_masks, _ = compute_blind_masks(data, channels, q)
    rows: List[Dict[str, object]] = []
    summary: Dict[str, Dict[str, float]] = {}
    for k in ks:
        kk = min(k, len(action_ids))
        per_ch_rates: Dict[str, List[float]] = {ch: [] for ch in channels}
        per_ch_jacc: Dict[str, List[float]] = {ch: [] for ch in channels}
        for b in range(int(n_boot)):
            sub_actions = rng.choice(action_ids, size=kk, replace=False)
            keep = np.isin(data["action_id"], sub_actions)
            subdata = {key: val[keep] if hasattr(val, "shape") and val.shape[:1] == data["world_delta"].shape[:1] else val for key, val in data.items()}
            _, masks, _ = compute_blind_masks(subdata, channels, q)
            for ch in channels:
                full_mask_sub = full_masks[ch][keep]
                rate = float(masks[ch].mean())
                jac = binary_jaccard(masks[ch], full_mask_sub)
                per_ch_rates[ch].append(rate)
                per_ch_jacc[ch].append(jac)
                rows.append({
                    "k": int(kk),
                    "bootstrap": int(b),
                    "channel": ch,
                    "blind_rate": rate,
                    "jaccard_vs_full_subset": jac,
                })
        row_summary: Dict[str, float] = {}
        for ch in channels:
            rates = np.asarray(per_ch_rates[ch], dtype=np.float32)
            jaccs = np.asarray(per_ch_jacc[ch], dtype=np.float32)
            row_summary[f"{ch}_blind_rate_mean"] = float(np.nanmean(rates))
            row_summary[f"{ch}_blind_rate_std"] = float(np.nanstd(rates))
            row_summary[f"{ch}_jaccard_mean"] = float(np.nanmean(jaccs))
            row_summary[f"{ch}_jaccard_std"] = float(np.nanstd(jaccs))
        summary[str(k)] = row_summary
    return rows, summary


def action_type_rows(data: Dict[str, np.ndarray], channels: Sequence[str], physical_nonnull: np.ndarray, blind_masks: Dict[str, np.ndarray]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    action_types = np.asarray(data["action_type"]).astype(str)
    for at in sorted(set(action_types.tolist())):
        keep = action_types == at
        row: Dict[str, object] = {
            "action_type": at,
            "n": int(keep.sum()),
            "world_delta_mean": float(np.mean(data["world_delta"][keep])) if keep.any() else float("nan"),
            "physical_nonnull_rate": float(np.mean(physical_nonnull[keep])) if keep.any() else float("nan"),
        }
        for ch in channels:
            row[f"detect_{ch}_mean"] = float(np.mean(data[f"detect_{ch}"][keep])) if keep.any() else float("nan")
            row[f"blind_{ch}_rate"] = float(np.mean(blind_masks[ch][keep])) if keep.any() else float("nan")
        rows.append(row)
    return rows


def leakage_probe_rows(data: Dict[str, np.ndarray], channels: Sequence[str], proj_dim: int, max_samples: int, seed: int) -> List[Dict[str, object]]:
    """Probe whether channel deltas expose action/outcome metadata.

    This is especially important for ``event_response`` because it is a
    post-action diagnostic response, not an observation available before a
    future prediction target.
    """
    if "delta_sample_indices" in data:
        dense_idx = np.asarray(data["delta_sample_indices"], dtype=np.int64)
    else:
        dense_idx = np.arange(data["world_delta"].shape[0], dtype=np.int64)
    sample_n = min(int(max_samples), int(len(dense_idx)))
    rng = np.random.default_rng(seed + 991)
    dense_rows = rng.choice(len(dense_idx), size=sample_n, replace=False)
    full_rows = dense_idx[dense_rows]

    action_types = np.asarray(data["action_type"]).astype(str)
    type_names = sorted(set(action_types.tolist()))
    type_to_id = {name: i for i, name in enumerate(type_names)}
    y_type_id = np.array([type_to_id[x] for x in action_types[full_rows]], dtype=np.int64)
    y_type = np.eye(len(type_names), dtype=np.float32)[y_type_id]
    y_action = np.asarray(data["action_array"][full_rows], dtype=np.float32)
    y_world = np.asarray(data["world_delta"][full_rows], dtype=np.float32)[:, None]

    rows: List[Dict[str, object]] = []
    embedded = _project_by_raw_shape(data, channels, dense_rows, proj_dim, seed + 313)
    for ch in channels:
        x = embedded[ch]
        type_pred, type_true = ridge_predict(x, y_type, alpha=1.0, seed=seed + 11, max_train=sample_n)
        type_acc = float(np.mean(np.argmax(type_pred, axis=1) == np.argmax(type_true, axis=1)))
        chance = float(1.0 / max(1, len(type_names)))
        action_r2 = ridge_r2(x, y_action, alpha=1.0, seed=seed + 13, max_train=sample_n)
        world_r2 = ridge_r2(x, y_world, alpha=1.0, seed=seed + 17, max_train=sample_n)
        rows.append(
            {
                "channel": ch,
                "target": "action_type",
                "score": type_acc,
                "chance": chance,
                "metric": "linear_ridge_accuracy",
            }
        )
        rows.append(
            {
                "channel": ch,
                "target": "action_array",
                "score": action_r2,
                "chance": 0.0,
                "metric": "linear_ridge_r2",
            }
        )
        rows.append(
            {
                "channel": ch,
                "target": "world_delta",
                "score": world_r2,
                "chance": 0.0,
                "metric": "linear_ridge_r2",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    data = normalize_schema(load_npz(args.data))
    channels = [str(x) for x in data["channels"].tolist()]
    dense_summary = validate_dense_deltas(data, channels)
    out = Path(args.out)
    report_dir = ensure_dir(out / "reports")
    fig_dir = ensure_dir(out / "figures")

    q = float(args.default_threshold)
    physical_nonnull, blind_masks, thresholds = compute_blind_masks(data, channels, q)
    blind_j = jaccard_matrix(blind_masks, channels)
    complementarity = 1.0 - blind_j
    np.fill_diagonal(complementarity, 0.0)

    redundancy = compute_redundancy_matrix(data, channels, proj_dim=args.projection_dim, max_samples=args.max_probe_samples, seed=args.seed)

    write_matrix_csv(report_dir / "blind_jaccard.csv", channels, blind_j)
    write_matrix_csv(report_dir / "complementarity.csv", channels, complementarity)
    write_matrix_csv(report_dir / "redundancy_r2.csv", channels, redundancy)

    plot_effect_histograms(fig_dir / "effect_histograms.png", data, channels)
    plot_examples(fig_dir / "examples.png", data, channels)
    plot_matrix(fig_dir / "blind_jaccard.png", channels, blind_j, "Blind-locus Jaccard", vmin=0, vmax=1)
    plot_matrix(fig_dir / "complementarity.png", channels, complementarity, "Blind-locus complementarity", vmin=0, vmax=1)
    plot_matrix(fig_dir / "redundancy_r2.png", channels, redundancy, "Redundancy probe R^2", vmin=-0.2, vmax=1)

    threshold_summaries = {}
    for qq in args.threshold_quantiles:
        _, masks_q, thresholds_q = compute_blind_masks(data, channels, float(qq))
        mat_q = jaccard_matrix(masks_q, channels)
        comp_q = 1.0 - mat_q
        np.fill_diagonal(comp_q, 0.0)
        tag = f"q{float(qq):.2f}".replace(".", "p")
        write_matrix_csv(report_dir / f"blind_jaccard_{tag}.csv", channels, mat_q)
        write_matrix_csv(report_dir / f"complementarity_{tag}.csv", channels, comp_q)
        write_matrix_csv(report_dir / f"threshold_jaccard_{tag}.csv", channels, mat_q)
        write_matrix_csv(report_dir / f"threshold_complementarity_{tag}.csv", channels, comp_q)
        threshold_summaries[str(qq)] = {
            "thresholds": thresholds_q,
            "blind_rate": {ch: float(masks_q[ch].mean()) for ch in channels},
            "mean_pairwise_jaccard": float(np.nanmean(mat_q[np.triu_indices(len(channels), k=1)])),
        }

    k_rows, k_summary = k_sweep_bootstrap_rows(data, channels, args.k_sweep, q=q, seed=args.seed, n_boot=args.k_bootstrap)
    write_long_csv(report_dir / "k_sweep_bootstrap.csv", k_rows)

    action_rows = action_type_rows(data, channels, physical_nonnull, blind_masks)
    write_long_csv(report_dir / "action_type_effects.csv", action_rows)

    leakage_rows = leakage_probe_rows(data, channels, args.projection_dim, args.max_probe_samples, args.seed)
    write_long_csv(report_dir / "leakage_probe.csv", leakage_rows)

    consistency_rows = control_consistency_rows(data)
    write_long_csv(report_dir / "rgb_control_consistency.csv", consistency_rows)

    blind_rates = {ch: float(blind_masks[ch].mean()) for ch in channels}
    summary = {
        "n_samples": int(data["world_delta"].shape[0]),
        "channels": channels,
        "schema_version": str(data.get("schema_version", np.array(["unknown"])).tolist()[0]),
        "diagnostic_only_channels": [str(x) for x in data.get("diagnostic_only_channels", np.array([])).tolist()],
        "stageb_default_channels": [str(x) for x in data.get("stageb_default_channels", np.array([])).tolist()],
        "default_threshold_quantile": q,
        "thresholds": thresholds,
        "physical_nonnull_rate": float(physical_nonnull.mean()),
        "blind_rates": blind_rates,
        "threshold_summaries": threshold_summaries,
        "k_sweep_bootstrap": k_summary,
        "dense_delta_summary": dense_summary,
        "leakage_probe": leakage_rows,
        "rgb_control_consistency": consistency_rows,
        "notes": [
            "Stage 0 is a channel adequacy test, not a final alignment result.",
            "Noisy RGB, gray RGB, and blur RGB are primary redundancy controls.",
            "Edge is a nonlinear derived-view diagnostic, not the primary redundancy control.",
            "Semantic is a privileged diagnostic channel, not a main modality.",
            "event_response is a diagnostic post-action event-response channel; do not use it as a future model input.",
        ],
    }
    save_json(summary, report_dir / "stage0_summary.json")
    print(f"Wrote Stage 0 reports to {out}")


if __name__ == "__main__":
    main()
