#!/usr/bin/env python
"""Generate Stage 0 counterfactual action-effect data.

The generated dataset stores, for each sampled state-action pair:

- action-induced physical change magnitude E_x;
- channel-wise observation detectability scores for all samples;
- dense channel deltas for a configurable subset of samples;
- optional pre-action/static observations for that same dense subset;
- metadata for action type and state/action IDs;
- example observations for plotting.

Dense deltas can be memory-heavy. For larger Stage 0 runs, use
``--max-delta-samples`` to store only a subset for redundancy probes while still
storing detectability scores for all state-action pairs.
"""

from __future__ import annotations

# Allow running scripts from a fresh clone before `pip install -e .`.
import sys
from pathlib import Path as _Path
_REPO_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from sae_align.envs.toy_powder import (
    Action,
    DEFAULT_CHANNELS,
    EVENT_NAMES,
    ToyPowderWorld,
    render_channel,
)
from sae_align.utils.io import ensure_dir, load_json


def world_effect_magnitude(grid_a: np.ndarray, grid_b: np.ndarray) -> float:
    return float(np.mean(grid_a != grid_b))


def channel_detectability(delta: np.ndarray) -> float:
    return float(np.mean(np.abs(delta.astype(np.float32))))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/stage0_toy.json")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n-states", type=int, default=None)
    p.add_argument("--k-actions", type=int, default=None)
    p.add_argument("--grid-size", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--channels", nargs="*", default=None)
    p.add_argument("--store-examples", type=int, default=16)
    p.add_argument(
        "--max-delta-samples",
        type=int,
        default=None,
        help="Store dense delta arrays for at most this many state-action samples. Detectability is still stored for all samples.",
    )
    p.add_argument(
        "--max-delta-bytes",
        type=int,
        default=None,
        help="Fail before generation if dense delta storage is estimated to exceed this many bytes.",
    )
    p.add_argument(
        "--store-static-obs",
        action="store_true",
        help="Also store pre-action/static observations as obs0_<channel> for the dense delta subset.",
    )
    p.add_argument(
        "--max-static-bytes",
        type=int,
        default=None,
        help="Fail before generation if static obs0 storage is estimated to exceed this many bytes.",
    )
    return p.parse_args()


def estimate_channel_storage_bytes(grid_size: int, channels: List[str], n_samples: int) -> int:
    probe_grid = np.zeros((grid_size, grid_size), dtype=np.int16)
    probe_action = Action("place", x=min(1, grid_size - 1), y=min(1, grid_size - 1), element=1)
    event_counts = np.zeros(len(EVENT_NAMES), dtype=np.float32)
    per_sample = 0
    for ch in channels:
        arr = render_channel(probe_grid, ch, action=probe_action, event_counts=event_counts, noise_seed=0)
        per_sample += int(arr.astype(np.float32).nbytes)
    return int(per_sample * n_samples)


def estimate_delta_bytes(grid_size: int, channels: List[str], max_delta_samples: int) -> int:
    return estimate_channel_storage_bytes(grid_size, channels, max_delta_samples)


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    n_states = args.n_states or int(cfg.get("n_states", 256))
    k_actions = args.k_actions or int(cfg.get("k_actions", 32))
    grid_size = args.grid_size or int(cfg.get("grid_size", 32))
    horizon = args.horizon or int(cfg.get("horizon", 3))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    channels = args.channels or list(cfg.get("channels", DEFAULT_CHANNELS))
    max_delta_samples = args.max_delta_samples
    if max_delta_samples is None:
        max_delta_samples = int(cfg.get("max_delta_samples", n_states * k_actions))
    max_delta_bytes = args.max_delta_bytes
    if max_delta_bytes is None:
        max_delta_bytes = int(cfg.get("max_delta_bytes", 1_000_000_000))
    store_static_obs = bool(args.store_static_obs or cfg.get("store_static_obs", False))
    max_static_bytes = args.max_static_bytes
    if max_static_bytes is None:
        max_static_bytes = int(cfg.get("max_static_bytes", 1_000_000_000))

    env = ToyPowderWorld(grid_size=grid_size, seed=seed)
    rng = np.random.default_rng(seed + 12345)
    states = [env.sample_state() for _ in range(n_states)]
    actions = env.make_action_bank(k=k_actions)
    noop = Action("noop")
    zero_events = np.zeros(len(EVENT_NAMES), dtype=np.float32)

    n = n_states * k_actions
    max_delta_samples = max(1, min(int(max_delta_samples), n))
    estimated_delta_bytes = estimate_delta_bytes(grid_size, channels, max_delta_samples)
    if estimated_delta_bytes > max_delta_bytes:
        raise MemoryError(
            "Estimated dense delta storage is too large: "
            f"{estimated_delta_bytes} bytes > budget {max_delta_bytes} bytes. "
            "Lower --max-delta-samples or raise --max-delta-bytes intentionally."
        )
    estimated_static_bytes = (
        estimate_channel_storage_bytes(grid_size, channels, max_delta_samples) if store_static_obs else 0
    )
    if store_static_obs and estimated_static_bytes > max_static_bytes:
        raise MemoryError(
            "Estimated static obs0 storage is too large: "
            f"{estimated_static_bytes} bytes > budget {max_static_bytes} bytes. "
            "Lower --max-delta-samples, disable --store-static-obs, or raise --max-static-bytes intentionally."
        )
    if max_delta_samples >= n:
        delta_keep = np.ones(n, dtype=bool)
    else:
        delta_keep = np.zeros(n, dtype=bool)
        delta_keep[rng.choice(n, size=max_delta_samples, replace=False)] = True

    world_delta = np.zeros(n, dtype=np.float32)
    state_id = np.zeros(n, dtype=np.int32)
    action_id = np.zeros(n, dtype=np.int32)
    action_type = np.empty(n, dtype="U16")
    action_array = np.zeros((n, 5), dtype=np.int16)

    detect: Dict[str, np.ndarray] = {ch: np.zeros(n, dtype=np.float32) for ch in channels}
    deltas: Dict[str, List[np.ndarray]] = {ch: [] for ch in channels}
    static_obs: Dict[str, List[np.ndarray]] = {ch: [] for ch in channels}

    example_obs = {ch: [] for ch in channels}
    example_meta = []

    idx = 0
    for si, grid in enumerate(states):
        noop_roll = env.rollout(grid, noop, horizon=horizon)
        for ai, action in enumerate(actions):
            act_roll = env.rollout(grid, action, horizon=horizon)
            world_delta[idx] = world_effect_magnitude(act_roll.final_grid, noop_roll.final_grid)
            state_id[idx] = si
            action_id[idx] = ai
            action_type[idx] = action.action_type
            action_array[idx] = action.as_array()

            for ch in channels:
                seed_base = int((si + 1) * 100000 + (ai + 7) * 997)
                if store_static_obs and delta_keep[idx]:
                    obs0 = render_channel(
                        grid,
                        ch,
                        action=action,
                        event_counts=zero_events,
                        noise_seed=seed_base,
                    )
                    static_obs[ch].append(obs0.astype(np.float32))
                obs_a = render_channel(
                    act_roll.final_grid,
                    ch,
                    action=action,
                    event_counts=act_roll.event_counts,
                    noise_seed=seed_base,
                )
                obs_n = render_channel(
                    noop_roll.final_grid,
                    ch,
                    action=action,
                    event_counts=noop_roll.event_counts,
                    noise_seed=seed_base,
                )
                delta = (obs_a - obs_n).astype(np.float32)
                detect[ch][idx] = channel_detectability(delta)
                if delta_keep[idx]:
                    deltas[ch].append(delta)
                if idx < args.store_examples:
                    example_obs[ch].append(obs_a.astype(np.float32))

            if idx < args.store_examples:
                example_meta.append((si, ai, action.action_type, int(action.x), int(action.y), int(action.element)))
            idx += 1

    arrays: Dict[str, np.ndarray] = {
        "world_delta": world_delta,
        "state_id": state_id,
        "action_id": action_id,
        "action_type": action_type,
        "action_array": action_array,
        "event_names": np.array(EVENT_NAMES),
        "channels": np.array(channels),
        "grid_size": np.array([grid_size], dtype=np.int32),
        "horizon": np.array([horizon], dtype=np.int32),
        "seed": np.array([seed], dtype=np.int32),
        "schema_version": np.array(["stage0_v2"]),
        "delta_sample_indices": np.nonzero(delta_keep)[0].astype(np.int64),
        "max_delta_samples": np.array([max_delta_samples], dtype=np.int32),
        "estimated_delta_bytes": np.array([estimated_delta_bytes], dtype=np.int64),
        "store_static_obs": np.array([store_static_obs]),
        "estimated_static_obs_bytes": np.array([estimated_static_bytes], dtype=np.int64),
        "diagnostic_only_channels": np.array(list(cfg.get("diagnostic_only_channels", ["event_response"]))),
        "stageb_default_channels": np.array(list(cfg.get("stageb_default_channels", ["rgb", "range", "local"]))),
    }
    if store_static_obs:
        arrays["static_obs_sample_indices"] = np.nonzero(delta_keep)[0].astype(np.int64)

    for ch in channels:
        arrays[f"detect_{ch}"] = detect[ch]
        if deltas[ch]:
            arrays[f"delta_{ch}"] = np.stack(deltas[ch], axis=0).astype(np.float32)
        if store_static_obs and static_obs[ch]:
            arrays[f"obs0_{ch}"] = np.stack(static_obs[ch], axis=0).astype(np.float32)
        if example_obs[ch]:
            arrays[f"example_{ch}"] = np.stack(example_obs[ch], axis=0).astype(np.float32)

    if example_meta:
        arrays["example_meta"] = np.array(example_meta, dtype=object)

    out = Path(args.out)
    ensure_dir(out.parent)
    np.savez_compressed(out, **arrays)
    print(
        f"Wrote {out} with {n} state-action samples, "
        f"delta_samples={int(delta_keep.sum())}, channels={channels}"
    )


if __name__ == "__main__":
    main()
