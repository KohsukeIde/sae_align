#!/usr/bin/env python
"""Create datasets for Action-IV task sanity and prototype experiments.

Backends:
  * synthetic: deterministic small data, no Powderworld dependency; for smoke tests.
  * powderworld: official Powderworld task generator, e.g. PWTaskGenDestroyAll.

The output schema is intentionally close to previous staged datasets:
  obs0_rgb, obs0_range, obs0_local
  delta_rgb, delta_range, delta_local
  detect_rgb, detect_range, detect_local
  action_array, action_type, state_id, action_id
  target_task_success, target_reward_delta, target_reward_after_action
  world_delta
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)


PALETTE = np.array([
    [236, 240, 241], [108, 122, 137], [243, 194, 58], [75, 119, 190],
    [179, 157, 219], [202, 105, 36], [137, 196, 244], [249, 104, 14],
    [38, 194, 129], [38, 67, 72], [157, 41, 51], [176, 207, 120],
    [255, 179, 167], [191, 85, 236], [0, 229, 255], [61, 90, 254],
    [121, 85, 72], [56, 142, 60], [158, 157, 36], [198, 40, 40],
    [224, 64, 251],
], dtype=np.float32) / 255.0


def render_rgb(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    ids = np.clip(ids, 0, PALETTE.shape[0] - 1)
    return PALETTE[ids].astype(np.float32)


def render_range(ids: np.ndarray) -> np.ndarray:
    """Cheap range-like channel: row/column normalized distances to non-empty cells."""
    ids = np.asarray(ids)
    solid = (ids != 0).astype(np.float32)
    h, w = solid.shape
    # Four cumulative visibility proxies, not a true lidar.
    left = np.cumsum(solid, axis=1) > 0
    right = np.cumsum(solid[:, ::-1], axis=1)[:, ::-1] > 0
    up = np.cumsum(solid, axis=0) > 0
    down = np.cumsum(solid[::-1, :], axis=0)[::-1, :] > 0
    out = np.stack([left, right, up, down], axis=-1).astype(np.float32)
    return out


def render_local(ids: np.ndarray, action: np.ndarray, radius: int = 3) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    h, w = ids.shape
    if action.size >= 5:
        # Action order used in this phase: xbin, ybin, elem, dx, dy.
        x = int(action[0]) * 8 + int(action[3])
        y = int(action[1]) * 8 + int(action[4])
    else:
        x = int(action[0]) % w
        y = int(action[1]) % h
    return render_local_xy(ids, x, y, radius=radius)


def render_local_xy(ids: np.ndarray, x: int, y: int, radius: int = 3) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    h, w = ids.shape
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    patch = np.zeros((2 * radius + 1, 2 * radius + 1, 1), dtype=np.float32)
    src = ids[y0:y1, x0:x1].astype(np.float32) / 20.0
    patch[(y0 - (y - radius)):(y1 - (y - radius)), (x0 - (x - radius)):(x1 - (x - radius)), 0] = src
    return patch


def detect(arr: np.ndarray) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float32).reshape(arr.shape[0], -1)
    return np.linalg.norm(flat, axis=1).astype(np.float32)


def make_synthetic(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    n_states = int(args.n_states)
    actions_per_state = int(args.actions_per_state)
    h = w = int(args.grid_size)
    n = n_states * actions_per_state

    states = rng.integers(0, 8, size=(n_states, h, w), dtype=np.int64)
    # Make empty dominate but not saturate.
    mask_empty = rng.random(size=states.shape) < 0.45
    states[mask_empty] = 0
    state_ids = np.repeat(np.arange(n_states, dtype=np.int64), actions_per_state)
    action_ids = np.tile(np.arange(actions_per_state, dtype=np.int64), n_states)

    action_array = np.zeros((n, 5), dtype=np.float32)
    action_type = np.empty(n, dtype="U16")
    target_delta = np.zeros(n, dtype=np.float32)
    target_after = np.zeros(n, dtype=np.float32)

    obs0_rgb, obs0_range, obs0_local = [], [], []
    delta_rgb, delta_range, delta_local = [], [], []
    world_delta = []

    action_bank = []
    for aid in range(actions_per_state):
        typ = ["place", "erase", "push"][int(aid % 3)]
        y = int(rng.integers(1, h - 1))
        x = int(rng.integers(1, w - 1))
        elem = int(rng.integers(1, 8))
        action_bank.append((typ, x, y, elem))

    for idx, (sid, aid) in enumerate(zip(state_ids, action_ids)):
        grid = states[int(sid)].copy()
        typ, x, y, elem = action_bank[int(aid)]
        action_type[idx] = typ
        action_array[idx] = np.array([x // 8, y // 8, elem, x % 8, y % 8], dtype=np.float32)

        next_grid = grid.copy()
        noop_grid = grid.copy()
        if typ == "place":
            next_grid[y-1:y+2, x-1:x+2] = elem
        elif typ == "erase":
            next_grid[y-1:y+2, x-1:x+2] = 0
        else:
            patch = next_grid[y-1:y+2, x-1:x+2].copy()
            next_grid[y-1:y+2, x-1:x+2] = np.roll(patch, shift=1, axis=1)

        # Synthetic task: positive when action increases empty space or creates fire near plant-like ids.
        reward0 = float(np.mean(noop_grid == 0))
        reward1 = float(np.mean(next_grid == 0))
        if elem == 7:
            yy0, yy1 = max(0, y - 2), min(h, y + 3)
            xx0, xx1 = max(0, x - 2), min(w, x + 3)
            local_plants = grid[yy0:yy1, xx0:xx1] == 8
            reward1 += 0.25 * float(np.mean(local_plants)) if local_plants.size else 0.0
        target_delta[idx] = reward1 - reward0
        target_after[idx] = reward1

        rgb0, rgb1 = render_rgb(grid), render_rgb(next_grid)
        ran0, ran1 = render_range(grid), render_range(next_grid)
        loc0, loc1 = render_local(grid, action_array[idx]), render_local(next_grid, action_array[idx])
        obs0_rgb.append(rgb0)
        obs0_range.append(ran0)
        obs0_local.append(loc0)
        delta_rgb.append(rgb1 - rgb0)
        delta_range.append(ran1 - ran0)
        delta_local.append(loc1 - loc0)
        world_delta.append((next_grid - noop_grid).astype(np.float32)[..., None])

    out = {
        "state_id": state_ids,
        "action_id": action_ids,
        "action_array": action_array,
        "action_type": action_type,
        "obs0_rgb": np.stack(obs0_rgb).astype(np.float32),
        "obs0_range": np.stack(obs0_range).astype(np.float32),
        "obs0_local": np.stack(obs0_local).astype(np.float32),
        "delta_rgb": np.stack(delta_rgb).astype(np.float32),
        "delta_range": np.stack(delta_range).astype(np.float32),
        "delta_local": np.stack(delta_local).astype(np.float32),
        "world_delta": np.stack(world_delta).astype(np.float32),
        "detect_rgb": detect(np.stack(delta_rgb)),
        "detect_range": detect(np.stack(delta_range)),
        "detect_local": detect(np.stack(delta_local)),
        "target_reward_delta": target_delta.astype(np.float32),
        "target_task_success": (target_delta > 1e-6).astype(np.float32),
        "target_reward_after_action": target_after.astype(np.float32),
        "metadata_json": np.array(json.dumps({"backend": "synthetic", "seed": args.seed}), dtype=object),
    }
    return out


def world_to_ids(world, num_elements: int) -> np.ndarray:
    """Convert Powderworld one-hot/density tensor to element ids."""
    ids = world[:, : int(num_elements)].argmax(dim=1)
    return ids.detach().cpu().numpy().astype(np.int64)


def simulate_steps(world, steps: int, pw, torch_mod, seed: int | None = None):
    w = world.clone()
    if seed is not None:
        torch_mod.manual_seed(int(seed))
    for _ in range(int(steps)):
        w = pw(w)
    return w


def destroy_reward(world, n_settle: int, pw, torch_mod, seed: int | None = None) -> np.ndarray:
    """Approximate installed PWDestroyEnv terminal reward on cloned worlds."""
    w = simulate_steps(world, n_settle, pw, torch_mod, seed=seed)
    return (w[:, 0:1].sum(dim=(1, 2, 3)).detach().cpu().numpy() / 1000.0).astype(np.float32)


def make_powderworld(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    try:
        import torch
        import powderworld.dists as pw_dists
        import powderworld.envs as pw_envs
        from powderworld.sim import PWSim
    except Exception as exc:  # pragma: no cover - depends on optional powderworld
        raise RuntimeError("Powderworld backend requires `pip install powderworld` or local install.") from exc

    if args.task_gen not in ("PWTaskGenDestroyAll", "PWDestroyEnv", "destroy"):
        raise ValueError(
            "The installed Powderworld package exposes PWDestroyEnv rather than "
            f"{args.task_gen!r}; use --task-gen PWDestroyEnv/PWTaskGenDestroyAll "
            "for the Action-IV task oracle sanity audit."
        )
    device = torch.device(args.device)
    pw = PWSim(device, bool(args.use_jit))
    world0 = torch.zeros((int(args.n_states), pw.NUM_CHANNEL, 64, 64), dtype=torch.float32, device=device)
    kwargs = dict(pw_envs.kwargs_pcg_default)
    np.random.seed(int(args.seed))
    for sid in range(int(args.n_states)):
        pw_dists.make_world(pw, world0[sid:sid+1], **{k: kwargs[k] for k in ["elems", "num_tasks", "num_lines", "num_circles", "num_squares"]})
    rng = np.random.default_rng(args.seed)
    action_resolution = 4
    hw = int(world0.shape[-1])

    n_states = int(args.n_states)
    actions_per_state = int(args.actions_per_state)
    n = n_states * actions_per_state
    state_ids = np.repeat(np.arange(n_states, dtype=np.int64), actions_per_state)
    action_ids = np.tile(np.arange(actions_per_state, dtype=np.int64), n_states)

    obs0_rgb, obs0_range, obs0_local = [], [], []
    delta_rgb, delta_range, delta_local = [], [], []
    world_delta = []
    action_array = np.zeros((n, 5), dtype=np.float32)
    action_type = np.empty(n, dtype="U16")
    target_delta = np.zeros(n, dtype=np.float32)
    target_reward_after = np.zeros(n, dtype=np.float32)

    # Fixed action bank across states for state-level signature compatibility.
    # Installed Powderworld VecEnv action order: xbin, ybin, elem, dxbin, dybin.
    elems = np.arange(0, min(21, int(pw.NUM_ELEMENTS)), dtype=np.int64)
    xbins = max(1, hw // action_resolution)
    ybins = max(1, hw // action_resolution)
    action_bank = []
    non_empty_elems = elems[elems != 0]
    for aid in range(actions_per_state):
        if non_empty_elems.size and rng.random() >= float(args.erase_action_fraction):
            elem = int(rng.choice(non_empty_elems))
        else:
            elem = 0
        action_bank.append(np.array([
            int(rng.integers(0, xbins)),
            int(rng.integers(0, ybins)),
            elem,
            int(rng.integers(0, 3)),
            int(rng.integers(0, 3)),
        ], dtype=np.float32))

    for idx, (sid, aid) in enumerate(zip(state_ids, action_ids)):
        action = action_bank[int(aid)].copy()
        action_array[idx] = action
        elem = int(action[2])
        action_type[idx] = "empty" if elem == 0 else "place"

        w0 = world0[int(sid):int(sid)+1].clone()
        noop = w0.clone()
        act = w0.clone()
        xbin, ybin, elem, dxbin, dybin = [int(v) for v in action.astype(np.int64)]
        real_x = xbin * action_resolution + action_resolution // 2
        real_y = ybin * action_resolution + action_resolution // 2
        dirx = dxbin - 1
        diry = dybin - 1
        radius = 5
        winddir = 20 * torch.tensor([dirx, diry], dtype=torch.float32, device=device)[None, :, None, None]
        pw.add_element(
            act[:, :, max(0, real_x - radius):min(hw, real_x + radius), max(0, real_y - radius):min(hw, real_y + radius)],
            int(elem),
            winddir,
        )
        rollout_seed = int(args.seed) * 1_000_003 + int(idx)
        noop = simulate_steps(noop, args.time_per_action, pw, torch, seed=rollout_seed)
        act = simulate_steps(act, args.time_per_action, pw, torch, seed=rollout_seed)
        ids0 = world_to_ids(w0, pw.NUM_ELEMENTS)[0]
        ids_noop = world_to_ids(noop, pw.NUM_ELEMENTS)[0]
        ids_act = world_to_ids(act, pw.NUM_ELEMENTS)[0]
        settle_seed = rollout_seed + 17
        reward0 = float(destroy_reward(noop, args.n_world_settle, pw, torch, seed=settle_seed)[0])
        reward1 = float(destroy_reward(act, args.n_world_settle, pw, torch, seed=settle_seed)[0])
        target_delta[idx] = reward1 - reward0
        target_reward_after[idx] = reward1

        rgb0, rgb1 = render_rgb(ids0), render_rgb(ids_act)
        ran0, ran1 = render_range(ids0), render_range(ids_act)
        loc0, loc1 = render_local_xy(ids0, real_x, real_y), render_local_xy(ids_act, real_x, real_y)
        obs0_rgb.append(rgb0)
        obs0_range.append(ran0)
        obs0_local.append(loc0)
        delta_rgb.append(rgb1 - rgb0)
        delta_range.append(ran1 - ran0)
        delta_local.append(loc1 - loc0)
        world_delta.append((ids_act - ids_noop).astype(np.float32)[..., None])

    out = {
        "state_id": state_ids,
        "action_id": action_ids,
        "action_array": action_array,
        "action_type": action_type,
        "obs0_rgb": np.stack(obs0_rgb).astype(np.float32),
        "obs0_range": np.stack(obs0_range).astype(np.float32),
        "obs0_local": np.stack(obs0_local).astype(np.float32),
        "delta_rgb": np.stack(delta_rgb).astype(np.float32),
        "delta_range": np.stack(delta_range).astype(np.float32),
        "delta_local": np.stack(delta_local).astype(np.float32),
        "world_delta": np.stack(world_delta).astype(np.float32),
        "detect_rgb": detect(np.stack(delta_rgb)),
        "detect_range": detect(np.stack(delta_range)),
        "detect_local": detect(np.stack(delta_local)),
        "target_reward_delta": target_delta.astype(np.float32),
        "target_task_success": (target_delta > 1e-6).astype(np.float32),
        "target_reward_after_action": target_reward_after.astype(np.float32),
        "metadata_json": np.array(json.dumps({
            "backend": "powderworld",
            "task_gen": args.task_gen,
            "seed": args.seed,
            "time_per_action": args.time_per_action,
            "n_world_settle": args.n_world_settle,
            "erase_action_fraction": args.erase_action_fraction,
            "powderworld_env": "PWDestroyEnv",
        }), dtype=object),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["synthetic", "powderworld"], default="synthetic")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-states", type=int, default=256)
    parser.add_argument("--actions-per-state", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-gen", default="PWTaskGenDestroyAll")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-jit", action="store_true")
    parser.add_argument("--time-per-action", type=int, default=2)
    parser.add_argument("--n-world-settle", type=int, default=64)
    parser.add_argument("--erase-action-fraction", type=float, default=0.5)
    args = parser.parse_args()

    if args.backend == "synthetic":
        data = make_synthetic(args)
    else:
        data = make_powderworld(args)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    np.savez_compressed(out_path, **data)
    world_norm = np.linalg.norm(data["world_delta"].reshape(data["world_delta"].shape[0], -1), axis=1)
    summary = {
        "backend": args.backend,
        "n_rows": int(data["state_id"].shape[0]),
        "n_states": int(np.unique(data["state_id"]).size),
        "positive_rate": float(np.mean(data["target_task_success"])),
        "reward_delta_mean": float(np.mean(data["target_reward_delta"])),
        "world_delta_mean": float(np.nan_to_num(np.mean(world_norm), nan=0.0)),
    }
    write_json(out_path.parent / "task_dataset_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
