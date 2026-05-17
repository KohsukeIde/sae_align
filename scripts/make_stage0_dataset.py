#!/usr/bin/env python
"""Generate Stage 0 counterfactual action-effect data.

The generated dataset stores, for each sampled state-action pair:

- action-induced physical change magnitude E_x;
- channel-wise observation deltas and detectability scores;
- metadata for action type and state/action IDs;
- example observations for plotting.

This script uses the toy Powderworld-like environment by default.  It is meant
as a protocol/debugging entrypoint, not the final simulator integration.
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    n_states = args.n_states or int(cfg.get("n_states", 256))
    k_actions = args.k_actions or int(cfg.get("k_actions", 32))
    grid_size = args.grid_size or int(cfg.get("grid_size", 32))
    horizon = args.horizon or int(cfg.get("horizon", 3))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    channels = args.channels or list(cfg.get("channels", DEFAULT_CHANNELS))

    env = ToyPowderWorld(grid_size=grid_size, seed=seed)
    rng = np.random.default_rng(seed)
    states = [env.sample_state() for _ in range(n_states)]
    actions = env.make_action_bank(k=k_actions)
    noop = Action("noop")

    n = n_states * k_actions
    world_delta = np.zeros(n, dtype=np.float32)
    state_id = np.zeros(n, dtype=np.int32)
    action_id = np.zeros(n, dtype=np.int32)
    action_type = np.empty(n, dtype="U16")
    action_array = np.zeros((n, 5), dtype=np.int16)

    detect: Dict[str, np.ndarray] = {ch: np.zeros(n, dtype=np.float32) for ch in channels}
    deltas: Dict[str, List[np.ndarray]] = {ch: [] for ch in channels}

    example_obs = {ch: [] for ch in channels}
    example_meta = []

    idx = 0
    for si, grid in enumerate(states):
        # No-op final grid per state can be reused for each action.
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
                deltas[ch].append(delta)
                detect[ch][idx] = channel_detectability(delta)
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
    }
    for ch in channels:
        arrays[f"detect_{ch}"] = detect[ch]
        arrays[f"delta_{ch}"] = np.stack(deltas[ch], axis=0).astype(np.float32)
        if example_obs[ch]:
            arrays[f"example_{ch}"] = np.stack(example_obs[ch], axis=0).astype(np.float32)
    if example_meta:
        arrays["example_meta"] = np.array(example_meta, dtype=object)

    out = Path(args.out)
    ensure_dir(out.parent)
    np.savez_compressed(out, **arrays)
    print(f"Wrote {out} with {n} state-action samples and channels={channels}")


if __name__ == "__main__":
    main()
