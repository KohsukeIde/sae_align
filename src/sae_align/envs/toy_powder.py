"""A lightweight Powderworld-like simulator for Stage 0 experiments.

This module intentionally implements only the minimal pieces needed for the
first diagnostic phase:

- random initial grids;
- deterministic local dynamics;
- intervention action bank;
- no-op vs do-action counterfactual rollouts;
- sensor-like observation channels with distinct blind loci.

It is not a faithful reimplementation of the original Powderworld environment.
Use it to debug the Stage 0 protocol before swapping in the real simulator via
``PowderworldAdapter``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

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


@dataclass(frozen=True)
class Action:
    """Discrete intervention action.

    action_type:
        One of ``place``, ``erase``, ``push``, or ``noop``.
    x, y:
        Grid coordinate.  Stored as x-column, y-row.
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

    # ---------------------------------------------------------------------
    # State generation and actions
    # ---------------------------------------------------------------------
    def sample_state(self) -> np.ndarray:
        """Sample a random grid state."""
        h = w = self.grid_size
        grid = np.zeros((h, w), dtype=np.int16)

        # Boundary walls.
        grid[0, :] = WALL
        grid[-1, :] = WALL
        grid[:, 0] = WALL
        grid[:, -1] = WALL

        # Random blobs for elements.  The probabilities are intentionally
        # sparse to leave room for dynamics.
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
            # Later elements only fill empty cells, preventing total collapse
            # into dense clutter.
            sub[(sub == EMPTY) & mask] = elem
            grid[inner] = sub

        return grid

    def make_action_bank(self, k: int = 128) -> List[Action]:
        """Create a fixed intervention bank.

        The bank is deliberately mixed across local placement, deletion, and
        push-like perturbations.  Downstream analyses should use K-sweeps over
        subsets of this bank to check action-sampling stability.
        """
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

    # ---------------------------------------------------------------------
    # Dynamics
    # ---------------------------------------------------------------------
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

        # Fire interactions.
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

        # Sand falls downward.  Iterate bottom-up for simple stability.
        for y in range(h - 2, 0, -1):
            for x in range(1, w - 1):
                if g[y, x] == SAND and g[y + 1, x] in (EMPTY, WATER):
                    g[y + 1, x], g[y, x] = g[y, x], g[y + 1, x]
                    events[EVENT_INDEX["sand_fall"]] += 1.0

        # Water flows down, otherwise sideways deterministically depending on x.
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

        # Smoke rises and sometimes disappears.
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


# -------------------------------------------------------------------------
# Observation channels
# -------------------------------------------------------------------------

RGB_COLORS = np.array(
    [
        [0.02, 0.02, 0.02],  # empty
        [0.35, 0.35, 0.35],  # wall
        [0.78, 0.70, 0.45],  # sand
        [0.05, 0.20, 0.85],  # water
        [1.00, 0.25, 0.05],  # fire
        [0.35, 0.18, 0.05],  # wood
        [0.65, 0.65, 0.70],  # metal
        [0.06, 0.08, 0.10],  # glass intentionally close to empty => RGB blind
        [0.45, 0.45, 0.48],  # smoke
    ],
    dtype=np.float32,
)

HARDNESS = np.array([0.0, 1.0, 0.20, 0.05, 0.0, 0.55, 0.90, 0.35, 0.0], dtype=np.float32)
TEMPERATURE = np.array([0.0, 0.0, 0.0, 0.10, 1.0, 0.05, 0.0, 0.0, 0.45], dtype=np.float32)
FLUIDITY = np.array([0.0, 0.0, 0.20, 1.0, 0.0, 0.0, 0.0, 0.0, 0.55], dtype=np.float32)
SOLIDITY = np.array([0.0, 1.0, 0.30, 0.0, 0.0, 0.70, 1.0, 0.40, 0.0], dtype=np.float32)


def render_rgb(grid: np.ndarray, noise_std: float = 0.0, seed: int | None = None) -> np.ndarray:
    img = RGB_COLORS[grid]
    # Return CHW.
    img = np.moveaxis(img, -1, 0).astype(np.float32)
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        img = np.clip(img + rng.normal(0, noise_std, img.shape).astype(np.float32), 0.0, 1.0)
    return img


def render_semantic(grid: np.ndarray) -> np.ndarray:
    return grid.astype(np.float32)[None, :, :]


def render_edge(grid: np.ndarray) -> np.ndarray:
    # Derived view from semantic/RGB boundaries.  This is deliberately a
    # negative control rather than a main modality.
    edge = np.zeros_like(grid, dtype=np.float32)
    edge[:, 1:] += grid[:, 1:] != grid[:, :-1]
    edge[1:, :] += grid[1:, :] != grid[:-1, :]
    edge = np.clip(edge, 0.0, 1.0)
    return edge[None, :, :]


def render_range_like(grid: np.ndarray) -> np.ndarray:
    """Four directional ranges to blockers.

    Glass is intentionally treated as transparent to this range-like channel,
    producing a blind locus distinct from RGB.  The implementation uses linear
    scans rather than per-pixel ray marching so Stage 0 remains fast.
    """
    h, w = grid.shape
    blockers = np.isin(grid, [WALL, WOOD, METAL])  # GLASS is not a range blocker.
    out = np.zeros((4, h, w), dtype=np.float32)
    maxdist = float(max(h, w))

    # up distance
    for x in range(w):
        last = -1
        for y in range(h):
            if blockers[y, x]:
                last = y
                out[0, y, x] = 0.0
            else:
                out[0, y, x] = (y - last) / maxdist
    # down distance
    for x in range(w):
        last = h
        for y in range(h - 1, -1, -1):
            if blockers[y, x]:
                last = y
                out[1, y, x] = 0.0
            else:
                out[1, y, x] = (last - y) / maxdist
    # left distance
    for y in range(h):
        last = -1
        for x in range(w):
            if blockers[y, x]:
                last = x
                out[2, y, x] = 0.0
            else:
                out[2, y, x] = (x - last) / maxdist
    # right distance
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
    """Agent/action-centric local material response channel.

    The channel is local around the action site.  It is not a global visual
    rendering.  This gives it blind loci distinct from RGB/range channels.
    """
    patch_size = 2 * radius + 1
    y0, x0 = int(action.y), int(action.x)
    padded = np.pad(grid, radius, mode="constant", constant_values=WALL)
    y = y0 + radius
    x = x0 + radius
    patch = padded[y - radius : y + radius + 1, x - radius : x + radius + 1]
    props = np.stack(
        [
            HARDNESS[patch],
            TEMPERATURE[patch],
            FLUIDITY[patch],
            SOLIDITY[patch],
        ],
        axis=0,
    ).astype(np.float32)
    return props


def render_global_event(event_counts: np.ndarray) -> np.ndarray:
    # Spatially marginalized event-statistics channel.  This is not called
    # audio and intentionally contains no spatial location.  Direct intervention
    # events (place/erase/push) are excluded to avoid leaking the chosen action;
    # the channel only reports endogenous interaction events.
    physics_event_count = 5  # fire_water, fire_wood, sand_fall, water_flow, smoke_rise
    return event_counts[:physics_event_count].astype(np.float32)


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
    if channel == "range":
        return render_range_like(grid)
    if channel == "local":
        if action is None:
            raise ValueError("local channel requires an action")
        return render_local_interaction(grid, action)
    if channel == "event":
        if event_counts is None:
            raise ValueError("event channel requires event_counts")
        return render_global_event(event_counts)
    if channel == "semantic":
        return render_semantic(grid)
    if channel == "edge":
        return render_edge(grid)
    raise KeyError(f"Unknown channel: {channel}")


DEFAULT_CHANNELS = ["rgb", "range", "local", "event", "semantic", "edge", "noisy_rgb"]
MAIN_CHANNELS = ["rgb", "range", "local", "event"]
CONTROL_CHANNELS = ["semantic", "edge", "noisy_rgb"]

