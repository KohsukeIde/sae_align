"""A lightweight Powderworld-like simulator for Stage 0 experiments.

This module intentionally implements only the minimal pieces needed for the
first diagnostic phase:

- random initial grids;
- deterministic local dynamics;
- intervention action bank;
- no-op vs do-action counterfactual rollouts;
- observation channels with distinct action-effect blind loci.

It is not a faithful reimplementation of the original Powderworld environment.
Use it to debug the Stage 0 protocol before swapping in the real simulator via
``PowderworldAdapter``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# Element IDs.
EMPTY = 0
WALL = 1
SAND = 2
WATER = 3
FIRE = 4
WOOD = 5
METAL = 6
GLASS = 7
SMOKE = 8

ELEMENT_NAMES = {
    EMPTY: "empty",
    WALL: "wall",
    SAND: "sand",
    WATER: "water",
    FIRE: "fire",
    WOOD: "wood",
    METAL: "metal",
    GLASS: "glass",
    SMOKE: "smoke",
}

EVENT_NAMES = [
    "fire_water",
    "fire_wood",
    "sand_fall",
    "water_flow",
    "smoke_rise",
    "push_move",
    "place",
    "erase",
]
EVENT_INDEX = {name: i for i, name in enumerate(EVENT_NAMES)}

# Only these are exposed by the Stage-0 event-response channel. Direct
# intervention events are intentionally excluded to reduce action leakage.
PHYSICS_EVENT_NAMES = EVENT_NAMES[:5]
PHYSICS_EVENT_COUNT = len(PHYSICS_EVENT_NAMES)


@dataclass(frozen=True)
class Action:
    """Discrete intervention action.

    action_type:
        One of ``place``, ``erase``, ``push``, or ``noop``.
    x, y:
        Grid coordinate. Stored as x-column, y-row.
    element:
        Element to place for ``place`` actions.
    dx, dy:
        Direction for ``push`` actions.
    """

    action_type: str
    x: int = 0
    y: int = 0
    element: int = EMPTY
    dx: int = 0
    dy: int = 0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.element, self.dx, self.dy], dtype=np.int16)


@dataclass
class RolloutResult:
    final_grid: np.ndarray
    event_counts: np.ndarray


class ToyPowderWorld:
    """Small deterministic grid simulator with Powderworld-like interactions."""

    def __init__(self, grid_size: int = 32, seed: int = 0):
        self.grid_size = int(grid_size)
        self.rng = np.random.default_rng(seed)

    def sample_state(self) -> np.ndarray:
        """Sample a random grid state."""
        h = w = self.grid_size
        grid = np.zeros((h, w), dtype=np.int16)

        grid[0, :] = WALL
        grid[-1, :] = WALL
        grid[:, 0] = WALL
        grid[:, -1] = WALL

        probs = [
            (WALL, 0.025),
            (SAND, 0.070),
            (WATER, 0.055),
            (WOOD, 0.050),
            (FIRE, 0.015),
            (METAL, 0.020),
            (GLASS, 0.025),
        ]
        inner = (slice(1, -1), slice(1, -1))
        for elem, p in probs:
            mask = self.rng.random((h - 2, w - 2)) < p
            sub = grid[inner]
            sub[(sub == EMPTY) & mask] = elem
            grid[inner] = sub

        return grid

    def make_action_bank(self, k: int = 128) -> List[Action]:
        """Create a fixed intervention bank."""
        actions: List[Action] = []
        h = w = self.grid_size
        place_elems = [SAND, WATER, FIRE, WOOD, METAL, GLASS]
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for _ in range(k):
            action_family = self.rng.choice(["place", "erase", "push"], p=[0.45, 0.25, 0.30])
            x = int(self.rng.integers(1, w - 1))
            y = int(self.rng.integers(1, h - 1))
            if action_family == "place":
                elem = int(self.rng.choice(place_elems))
                actions.append(Action("place", x=x, y=y, element=elem))
            elif action_family == "erase":
                actions.append(Action("erase", x=x, y=y))
            else:
                dx, dy = dirs[int(self.rng.integers(0, len(dirs)))]
                actions.append(Action("push", x=x, y=y, dx=dx, dy=dy))
        return actions

    def apply_action(self, grid: np.ndarray, action: Action) -> Tuple[np.ndarray, np.ndarray]:
        grid = grid.copy()
        events = np.zeros(len(EVENT_NAMES), dtype=np.float32)
        if action.action_type == "noop":
            return grid, events

        x, y = int(action.x), int(action.y)
        if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]):
            return grid, events

        if action.action_type == "place":
            grid[y, x] = int(action.element)
            events[EVENT_INDEX["place"]] += 1.0
        elif action.action_type == "erase":
            if grid[y, x] != WALL:
                grid[y, x] = EMPTY
                events[EVENT_INDEX["erase"]] += 1.0
        elif action.action_type == "push":
            nx, ny = x + int(action.dx), y + int(action.dy)
            if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                if grid[y, x] not in (EMPTY, WALL) and grid[ny, nx] == EMPTY:
                    grid[ny, nx] = grid[y, x]
                    grid[y, x] = EMPTY
                    events[EVENT_INDEX["push_move"]] += 1.0
        return grid, events

    def step(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """One deterministic update step."""
        g = grid.copy()
        h, w = g.shape
        events = np.zeros(len(EVENT_NAMES), dtype=np.float32)

        new_g = g.copy()
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if g[y, x] != FIRE:
                    continue
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if g[ny, nx] == WATER:
                        new_g[y, x] = SMOKE
                        new_g[ny, nx] = SMOKE
                        events[EVENT_INDEX["fire_water"]] += 1.0
                    elif g[ny, nx] == WOOD:
                        new_g[ny, nx] = FIRE
                        events[EVENT_INDEX["fire_wood"]] += 1.0
        g = new_g

        for y in range(h - 2, 0, -1):
            for x in range(1, w - 1):
                if g[y, x] == SAND and g[y + 1, x] in (EMPTY, WATER):
                    g[y + 1, x], g[y, x] = g[y, x], g[y + 1, x]
                    events[EVENT_INDEX["sand_fall"]] += 1.0

        for y in range(h - 2, 0, -1):
            for x in range(1, w - 1):
                if g[y, x] != WATER:
                    continue
                if g[y + 1, x] == EMPTY:
                    g[y + 1, x], g[y, x] = WATER, EMPTY
                    events[EVENT_INDEX["water_flow"]] += 1.0
                else:
                    direction = -1 if (x + y) % 2 == 0 else 1
                    if g[y, x + direction] == EMPTY:
                        g[y, x + direction], g[y, x] = WATER, EMPTY
                        events[EVENT_INDEX["water_flow"]] += 1.0

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if g[y, x] == SMOKE:
                    if g[y - 1, x] == EMPTY:
                        g[y - 1, x], g[y, x] = SMOKE, EMPTY
                        events[EVENT_INDEX["smoke_rise"]] += 1.0
                    elif (x * 17 + y * 31) % 11 == 0:
                        g[y, x] = EMPTY

        return g, events

    def rollout(self, grid: np.ndarray, action: Action, horizon: int = 3) -> RolloutResult:
        g, action_events = self.apply_action(grid, action)
        total_events = action_events.copy()
        for _ in range(int(horizon)):
            g, ev = self.step(g)
            total_events += ev
        return RolloutResult(final_grid=g, event_counts=total_events)


RGB_COLORS = np.array(
    [
        [0.02, 0.02, 0.02],
        [0.35, 0.35, 0.35],
        [0.78, 0.70, 0.45],
        [0.05, 0.20, 0.85],
        [1.00, 0.25, 0.05],
        [0.35, 0.18, 0.05],
        [0.65, 0.65, 0.70],
        [0.06, 0.08, 0.10],
        [0.45, 0.45, 0.48],
    ],
    dtype=np.float32,
)

HARDNESS = np.array([0.0, 1.0, 0.20, 0.05, 0.0, 0.55, 0.90, 0.35, 0.0], dtype=np.float32)
TEMPERATURE = np.array([0.0, 0.0, 0.0, 0.10, 1.0, 0.05, 0.0, 0.0, 0.45], dtype=np.float32)
FLUIDITY = np.array([0.0, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0, 0.0, 0.55], dtype=np.float32)
SOLIDITY = np.array([0.0, 1.0, 0.30, 0.0, 0.0, 0.70, 1.0, 0.40, 0.0], dtype=np.float32)


def render_rgb(grid: np.ndarray, noise_std: float = 0.0, seed: int | None = None) -> np.ndarray:
    img = np.moveaxis(RGB_COLORS[grid], -1, 0).astype(np.float32)
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        img = np.clip(img + rng.normal(0, noise_std, img.shape).astype(np.float32), 0.0, 1.0)
    return img


def render_gray_rgb(grid: np.ndarray) -> np.ndarray:
    """Linear RGB-derived grayscale redundancy control."""
    rgb = render_rgb(grid)
    gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return gray[None, :, :].astype(np.float32)


def _mean_blur_chw(x: np.ndarray) -> np.ndarray:
    pad = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
    out = np.zeros_like(x, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += pad[:, dy : dy + x.shape[1], dx : dx + x.shape[2]]
    return out / 9.0


def render_blur_rgb(grid: np.ndarray) -> np.ndarray:
    """RGB-derived blurred redundancy control."""
    return _mean_blur_chw(render_rgb(grid)).astype(np.float32)


def render_semantic(grid: np.ndarray) -> np.ndarray:
    return grid.astype(np.float32)[None, :, :]


def render_edge(grid: np.ndarray) -> np.ndarray:
    """Boundary-derived diagnostic channel.

    This deterministic derived view isolates boundary changes rather than RGB
    color changes. It is therefore not the primary redundancy control.
    """
    edge = np.zeros_like(grid, dtype=np.float32)
    edge[:, 1:] += grid[:, 1:] != grid[:, :-1]
    edge[1:, :] += grid[1:, :] != grid[:-1, :]
    return np.clip(edge, 0.0, 1.0)[None, :, :]


def render_range_like(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    blockers = np.isin(grid, [WALL, WOOD, METAL])
    out = np.zeros((4, h, w), dtype=np.float32)
    maxdist = float(max(h, w))

    for x in range(w):
        last = -1
        for y in range(h):
            if blockers[y, x]:
                last = y
                out[0, y, x] = 0.0
            else:
                out[0, y, x] = (y - last) / maxdist

    for x in range(w):
        last = h
        for y in range(h - 1, -1, -1):
            if blockers[y, x]:
                last = y
                out[1, y, x] = 0.0
            else:
                out[1, y, x] = (last - y) / maxdist

    for y in range(h):
        last = -1
        for x in range(w):
            if blockers[y, x]:
                last = x
                out[2, y, x] = 0.0
            else:
                out[2, y, x] = (x - last) / maxdist

    for y in range(h):
        last = w
        for x in range(w - 1, -1, -1):
            if blockers[y, x]:
                last = x
                out[3, y, x] = 0.0
            else:
                out[3, y, x] = (last - x) / maxdist

    return np.clip(out, 0.0, 1.0)


def render_local_interaction(grid: np.ndarray, action: Action, radius: int = 3) -> np.ndarray:
    y0, x0 = int(action.y), int(action.x)
    padded = np.pad(grid, radius, mode="constant", constant_values=WALL)
    y = y0 + radius
    x = x0 + radius
    patch = padded[y - radius : y + radius + 1, x - radius : x + radius + 1]
    return np.stack(
        [HARDNESS[patch], TEMPERATURE[patch], FLUIDITY[patch], SOLIDITY[patch]],
        axis=0,
    ).astype(np.float32)


def render_event_response(event_counts: np.ndarray) -> np.ndarray:
    """Spatially marginalized post-action endogenous event response.

    This is a Stage-0 diagnostic response channel. It should not be used as a
    future world-model input without redesigning it into a pre-action event
    history channel.
    """
    return event_counts[:PHYSICS_EVENT_COUNT].astype(np.float32)


def render_channel(
    grid: np.ndarray,
    channel: str,
    action: Action | None = None,
    event_counts: np.ndarray | None = None,
    noise_seed: int | None = None,
) -> np.ndarray:
    if channel == "rgb":
        return render_rgb(grid, noise_std=0.0)
    if channel == "noisy_rgb":
        return render_rgb(grid, noise_std=0.05, seed=noise_seed)
    if channel == "gray_rgb":
        return render_gray_rgb(grid)
    if channel == "blur_rgb":
        return render_blur_rgb(grid)
    if channel == "range":
        return render_range_like(grid)
    if channel == "local":
        if action is None:
            raise ValueError("local channel requires an action")
        return render_local_interaction(grid, action)
    if channel in {"event", "event_response"}:
        if event_counts is None:
            raise ValueError("event_response channel requires event_counts")
        return render_event_response(event_counts)
    if channel == "semantic":
        return render_semantic(grid)
    if channel == "edge":
        return render_edge(grid)
    raise KeyError(f"Unknown channel: {channel}")


DEFAULT_CHANNELS = [
    "rgb",
    "range",
    "local",
    "event_response",
    "semantic",
    "edge",
    "noisy_rgb",
    "gray_rgb",
    "blur_rgb",
]
MAIN_CHANNELS = ["rgb", "range", "local", "event_response"]
CONTROL_CHANNELS = ["semantic", "edge", "noisy_rgb", "gray_rgb", "blur_rgb"]
