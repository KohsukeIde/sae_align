#!/usr/bin/env python
"""Train NumPy Stage B encoders from Stage 0 dense action-effect or static features."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from sae_align.models import StandardizedPCAEncoder, default_stageb_channels, save_transition_encoders
from sae_align.analysis.strata import diagnostic_only_channels, validate_dense_stage0_data, validate_dense_static_data
from sae_align.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--channels", nargs="*", default=None)
    p.add_argument("--n-components", type=int, default=32)
    p.add_argument("--max-train-samples", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--feature-kind",
        choices=["action-effect", "delta", "static", "obs0"],
        default="action-effect",
        help="Train on dense delta_<channel> action effects or obs0_<channel> static observations.",
    )
    p.add_argument("--no-redundancy-controls", action="store_true")
    p.add_argument("--include-diagnostic", action="store_true")
    p.add_argument("--include-event-response", action="store_true")
    p.add_argument(
        "--allow-leakage-diagnostic",
        action="store_true",
        help="Allow diagnostic-only post-action channels such as event_response. Results must not be used as primary Stage B evidence.",
    )
    return p.parse_args()


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def metadata_defaults(data: Dict[str, np.ndarray]) -> list[str] | None:
    if "stageb_default_channels" not in data:
        return None
    return [str(x) for x in data["stageb_default_channels"].tolist()]


def normalize_feature_kind(value: str) -> str:
    if value in {"action-effect", "delta"}:
        return "action-effect"
    if value in {"static", "obs0"}:
        return "static"
    raise ValueError(f"Unsupported feature kind: {value}")


def feature_prefix(feature_kind: str) -> str:
    return "delta" if feature_kind == "action-effect" else "obs0"


def choose_channels(args: argparse.Namespace, data: Dict[str, np.ndarray]) -> list[str]:
    kind = normalize_feature_kind(args.feature_kind)
    prefix = feature_prefix(kind)
    available = [str(x) for x in data["channels"].tolist()]
    if args.channels:
        channels = [str(x) for x in args.channels]
    else:
        channels = default_stageb_channels(
            available,
            metadata_defaults=metadata_defaults(data),
            include_redundancy_controls=not args.no_redundancy_controls,
            include_diagnostic=bool(args.include_diagnostic),
            include_event_response=bool(args.include_event_response),
        )
    missing = [ch for ch in channels if f"{prefix}_{ch}" not in data]
    if missing:
        raise KeyError(f"Missing dense {prefix} arrays for channels: {missing}")
    if not channels:
        raise ValueError("No Stage B channels selected.")
    leakage = sorted(set(channels) & diagnostic_only_channels(data))
    if kind == "action-effect" and leakage and not args.allow_leakage_diagnostic:
        raise ValueError(
            "Diagnostic-only post-action channels are excluded from primary Stage B: "
            f"{leakage}. Re-run with --allow-leakage-diagnostic only for separately labeled diagnostics."
        )
    return channels


def main() -> None:
    args = parse_args()
    data = load_npz(args.data)
    kind = normalize_feature_kind(args.feature_kind)
    prefix = feature_prefix(kind)
    channels = choose_channels(args, data)
    out = ensure_dir(args.out)
    rng = np.random.default_rng(args.seed)

    if kind == "action-effect":
        sample_indices = validate_dense_stage0_data(data, channels, require_detect=True)
    else:
        sample_indices = validate_dense_static_data(data, channels)
    n_dense = int(sample_indices.shape[0])
    train_idx = np.arange(n_dense)
    if n_dense > int(args.max_train_samples):
        train_idx = np.sort(rng.choice(n_dense, size=int(args.max_train_samples), replace=False))

    encoders: Dict[str, StandardizedPCAEncoder] = {}
    embeddings: Dict[str, np.ndarray] = {
        "channels": np.array(channels),
        "train_dense_indices": train_idx.astype(np.int64),
        "sample_indices": sample_indices.astype(np.int64),
    }

    for ch in channels:
        x_all = data[f"{prefix}_{ch}"]
        x_train = x_all[train_idx]
        encoder = StandardizedPCAEncoder(n_components=int(args.n_components)).fit(x_train)
        encoders[ch] = encoder
        embeddings[f"embedding_{ch}"] = encoder.transform(x_all)

    model_path = out / "transition_encoders.npz"
    metadata = {
        "data": str(Path(args.data)),
        "channels": channels,
        "feature_kind": kind,
        "feature_prefix": prefix,
        "n_dense_samples": n_dense,
        "n_train_samples": int(train_idx.shape[0]),
        "n_components_requested": int(args.n_components),
        "seed": int(args.seed),
        "include_diagnostic": bool(args.include_diagnostic),
        "include_event_response": bool(args.include_event_response),
        "allow_leakage_diagnostic": bool(args.allow_leakage_diagnostic),
    }
    save_transition_encoders(str(model_path), encoders, metadata=metadata)
    np.savez_compressed(out / "transition_embeddings.npz", **embeddings)
    save_json(metadata, out / "transition_encoder_metadata.json")
    print(f"Wrote Stage B transition encoders to {model_path}")


if __name__ == "__main__":
    main()
