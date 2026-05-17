#!/usr/bin/env python
"""Analyze Stage 0 observation-channel adequacy.

Outputs:
- blind-locus overlap matrices;
- complementarity matrices;
- redundancy-probe R^2 matrices;
- effect histograms and control plots;
- JSON summary including Go/No-go diagnostics.
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

from sae_align.analysis.metrics import binary_jaccard, ridge_r2
from sae_align.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--k-sweep", nargs="*", type=int, default=[16, 32, 64])
    p.add_argument("--threshold-quantiles", nargs="*", type=float, default=[0.05, 0.10, 0.20])
    p.add_argument("--default-threshold", type=float, default=0.10)
    p.add_argument("--projection-dim", type=int, default=128)
    p.add_argument("--max-probe-samples", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def write_matrix_csv(path: Path, names: Sequence[str], mat: np.ndarray) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + list(names))
        for name, row in zip(names, mat):
            writer.writerow([name] + [f"{x:.6f}" if np.isfinite(x) else "nan" for x in row])


def plot_matrix(path: Path, names: Sequence[str], mat: np.ndarray, title: str, vmin: float = 0.0, vmax: float = 1.0) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7, 6))
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


def compute_blind_masks(data: Dict[str, np.ndarray], channels: Sequence[str], q: float) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    wx = data["world_delta"]
    eps_x = float(np.quantile(wx, q))
    # Ensure tiny nonzero lower bound so null isn't all zero if q falls at 0.
    eps_x = max(eps_x, 1e-8)
    physical_nonnull = wx >= eps_x
    masks: Dict[str, np.ndarray] = {}
    thresholds = {"world": eps_x}
    for ch in channels:
        det = data[f"detect_{ch}"]
        # Channel thresholds are computed within the physical-nonnull subset.
        # Otherwise physical null samples dominate the lower quantiles and make
        # all modality-blind rates collapse to zero.
        det_base = det[physical_nonnull]
        if len(det_base) == 0:
            tau = 1e-8
        else:
            tau = float(np.quantile(det_base, q))
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
    # Display first example for image-like channels. Event/local are shown as heatmaps.
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


def compute_redundancy_matrix(data: Dict[str, np.ndarray], channels: Sequence[str], proj_dim: int, max_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = data["world_delta"].shape[0]
    if n > max_samples:
        sample = rng.choice(n, size=max_samples, replace=False)
    else:
        sample = np.arange(n)
    embedded: Dict[str, np.ndarray] = {}
    projectors: Dict[int, np.ndarray] = {}
    for i, ch in enumerate(channels):
        x = data[f"delta_{ch}"][sample]
        x2 = x.reshape(x.shape[0], -1).astype(np.float32)
        x2 = x2 - x2.mean(axis=0, keepdims=True)
        x2 = x2 / (x2.std(axis=0, keepdims=True) + 1e-6)
        in_dim = x2.shape[1]
        if in_dim <= proj_dim:
            embedded[ch] = x2
            continue
        # Use the same projection for channels with the same raw shape.  This
        # keeps redundancy controls such as rgb/noisy_rgb from being obscured by
        # unrelated random compressions.
        if in_dim not in projectors:
            proj_rng = np.random.default_rng(seed + 1000 + in_dim)
            projectors[in_dim] = proj_rng.normal(
                0.0,
                1.0 / np.sqrt(proj_dim),
                size=(in_dim, proj_dim),
            ).astype(np.float32)
        embedded[ch] = x2 @ projectors[in_dim]
    mat = np.zeros((len(channels), len(channels)), dtype=np.float32)
    for i, src in enumerate(channels):
        for j, tgt in enumerate(channels):
            if src == tgt:
                mat[i, j] = 1.0
            else:
                mat[i, j] = ridge_r2(embedded[src], embedded[tgt], alpha=1.0, seed=seed + 17 * i + j, max_train=max_samples)
    return mat


def k_sweep_summary(data: Dict[str, np.ndarray], channels: Sequence[str], ks: Sequence[int], q: float, seed: int) -> Dict[str, Dict[str, float]]:
    # This approximates action-subset stability by choosing the first k unique
    # action IDs from random subsets and recomputing blind prevalence.
    rng = np.random.default_rng(seed)
    action_ids = np.unique(data["action_id"])
    summary: Dict[str, Dict[str, float]] = {}
    _, full_masks, _ = compute_blind_masks(data, channels, q)
    for k in ks:
        kk = min(k, len(action_ids))
        sub_actions = rng.choice(action_ids, size=kk, replace=False)
        keep = np.isin(data["action_id"], sub_actions)
        subdata = {key: val[keep] if hasattr(val, "shape") and val.shape[:1] == data["world_delta"].shape[:1] else val for key, val in data.items()}
        _, masks, _ = compute_blind_masks(subdata, channels, q)
        row = {}
        for ch in channels:
            # Lift back to full coordinate for rough agreement.
            full_mask_sub = full_masks[ch][keep]
            row[f"{ch}_blind_rate"] = float(masks[ch].mean())
            row[f"{ch}_jaccard_vs_full_subset"] = binary_jaccard(masks[ch], full_mask_sub)
        summary[str(k)] = row
    return summary


def main() -> None:
    args = parse_args()
    data = load_npz(args.data)
    channels = [str(x) for x in data["channels"].tolist()]
    out = Path(args.out)
    report_dir = ensure_dir(out / "reports")
    fig_dir = ensure_dir(out / "figures")

    q = float(args.default_threshold)
    physical_nonnull, blind_masks, thresholds = compute_blind_masks(data, channels, q)
    blind_j = jaccard_matrix(blind_masks, channels)
    complementarity = 1.0 - blind_j
    np.fill_diagonal(complementarity, 0.0)

    redundancy = compute_redundancy_matrix(
        data,
        channels,
        proj_dim=args.projection_dim,
        max_samples=args.max_probe_samples,
        seed=args.seed,
    )

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
        threshold_summaries[str(qq)] = {
            "thresholds": thresholds_q,
            "blind_rate": {ch: float(masks_q[ch].mean()) for ch in channels},
            "mean_pairwise_jaccard": float(np.nanmean(mat_q[np.triu_indices(len(channels), k=1)])),
        }

    k_summary = k_sweep_summary(data, channels, args.k_sweep, q=q, seed=args.seed)

    blind_rates = {ch: float(blind_masks[ch].mean()) for ch in channels}
    summary = {
        "n_samples": int(data["world_delta"].shape[0]),
        "channels": channels,
        "default_threshold_quantile": q,
        "thresholds": thresholds,
        "physical_nonnull_rate": float(physical_nonnull.mean()),
        "blind_rates": blind_rates,
        "threshold_summaries": threshold_summaries,
        "k_sweep": k_summary,
        "notes": [
            "Stage 0 is a channel adequacy test, not a final alignment result.",
            "Edge and noisy_rgb should behave as negative/redundancy controls.",
            "Semantic is a privileged diagnostic channel, not a main modality.",
        ],
    }
    save_json(summary, report_dir / "stage0_summary.json")
    print(f"Wrote Stage 0 reports to {out}")


if __name__ == "__main__":
    main()
