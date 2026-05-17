from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_STAGEB_CHANNELS = ("rgb", "range", "local")
REDUNDANCY_CONTROL_CHANNELS = ("noisy_rgb", "gray_rgb", "blur_rgb")
DIAGNOSTIC_CHANNELS = ("event", "event_response", "semantic", "edge")


def _as_channel_list(values: Iterable[object]) -> list[str]:
    return [str(x) for x in values]


def default_stageb_channels(
    available_channels: Sequence[str],
    *,
    metadata_defaults: Sequence[str] | None = None,
    include_redundancy_controls: bool = True,
    include_diagnostic: bool = False,
    include_event_response: bool = False,
) -> list[str]:
    """Choose Stage B channels from an available Stage 0 dataset.

    The default is intentionally non-leaking: trainable channels are RGB, range,
    and local interaction views plus RGB-derived redundancy controls when
    present. Post-action event-response channels stay excluded unless explicitly
    requested.
    """
    available = list(dict.fromkeys(_as_channel_list(available_channels)))
    available_set = set(available)
    base = _as_channel_list(metadata_defaults or DEFAULT_STAGEB_CHANNELS)
    wanted = [ch for ch in base if ch in available_set]
    if include_redundancy_controls:
        wanted.extend(ch for ch in REDUNDANCY_CONTROL_CHANNELS if ch in available_set)
    if include_diagnostic:
        wanted.extend(ch for ch in DIAGNOSTIC_CHANNELS if ch in available_set)
    elif include_event_response:
        wanted.extend(ch for ch in ("event_response", "event") if ch in available_set)
    return list(dict.fromkeys(wanted))


@dataclass
class StandardizedPCAEncoder:
    """Small NumPy transition encoder for flattened action-effect deltas."""

    n_components: int = 32
    eps: float = 1e-6
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    components_: np.ndarray | None = None
    explained_variance_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardizedPCAEncoder":
        x2 = self._flatten(x)
        if x2.shape[0] < 2:
            raise ValueError("StandardizedPCAEncoder requires at least two samples.")
        self.mean_ = x2.mean(axis=0, keepdims=True).astype(np.float32)
        self.scale_ = (x2.std(axis=0, keepdims=True) + float(self.eps)).astype(np.float32)
        z = (x2 - self.mean_) / self.scale_
        z = z - z.mean(axis=0, keepdims=True)
        _, s, vt = np.linalg.svd(z, full_matrices=False)
        dim = min(int(self.n_components), vt.shape[0])
        self.components_ = vt[:dim].astype(np.float32)
        denom = max(1, z.shape[0] - 1)
        self.explained_variance_ = ((s[:dim] ** 2) / denom).astype(np.float32)
        return self

    def transform(self, x: np.ndarray, *, l2_normalize: bool = True) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.components_ is None:
            raise ValueError("Encoder is not fitted.")
        x2 = self._flatten(x)
        if x2.shape[1] != self.mean_.shape[1]:
            raise ValueError(f"Expected {self.mean_.shape[1]} features, got {x2.shape[1]}.")
        z = (x2 - self.mean_) / self.scale_
        emb = (z @ self.components_.T).astype(np.float32)
        if l2_normalize:
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.maximum(norm, self.eps)
        return emb

    def fit_transform(self, x: np.ndarray, *, l2_normalize: bool = True) -> np.ndarray:
        return self.fit(x).transform(x, l2_normalize=l2_normalize)

    @staticmethod
    def _flatten(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim < 2:
            raise ValueError("Expected an array with sample dimension and features.")
        return x.reshape(x.shape[0], -1)


def save_transition_encoders(
    path: str,
    encoders: Mapping[str, StandardizedPCAEncoder],
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    arrays: Dict[str, np.ndarray] = {
        "channels": np.array(list(encoders.keys())),
        "metadata_json": np.array([json.dumps(dict(metadata or {}), sort_keys=True)]),
    }
    for ch, encoder in encoders.items():
        if encoder.mean_ is None or encoder.scale_ is None or encoder.components_ is None:
            raise ValueError(f"Encoder for {ch} is not fitted.")
        arrays[f"{ch}__mean"] = encoder.mean_.astype(np.float32)
        arrays[f"{ch}__scale"] = encoder.scale_.astype(np.float32)
        arrays[f"{ch}__components"] = encoder.components_.astype(np.float32)
        arrays[f"{ch}__explained_variance"] = (
            encoder.explained_variance_.astype(np.float32)
            if encoder.explained_variance_ is not None
            else np.zeros(encoder.components_.shape[0], dtype=np.float32)
        )
        arrays[f"{ch}__n_components"] = np.array([encoder.components_.shape[0]], dtype=np.int32)
        arrays[f"{ch}__eps"] = np.array([encoder.eps], dtype=np.float32)
    np.savez_compressed(path, **arrays)


def load_transition_encoders(path: str) -> Tuple[Dict[str, StandardizedPCAEncoder], Dict[str, object]]:
    encoders: Dict[str, StandardizedPCAEncoder] = {}
    with np.load(path, allow_pickle=False) as f:
        channels = _as_channel_list(f["channels"].tolist())
        metadata = json.loads(str(f["metadata_json"][0])) if "metadata_json" in f else {}
        for ch in channels:
            encoders[ch] = StandardizedPCAEncoder(
                n_components=int(f[f"{ch}__n_components"][0]),
                eps=float(f[f"{ch}__eps"][0]),
                mean_=f[f"{ch}__mean"].astype(np.float32),
                scale_=f[f"{ch}__scale"].astype(np.float32),
                components_=f[f"{ch}__components"].astype(np.float32),
                explained_variance_=f[f"{ch}__explained_variance"].astype(np.float32),
            )
    return encoders, metadata
