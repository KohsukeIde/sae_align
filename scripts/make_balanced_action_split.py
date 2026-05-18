#!/usr/bin/env python
"""Create and audit a balanced probe/held-out action split for Stage B.3."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import argparse
import csv
from pathlib import Path
from typing import Mapping

import numpy as np

from sae_align.utils.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--probe-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-candidates", type=int, default=5000)
    return p.parse_args()


def load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def mode_str(values: np.ndarray) -> str:
    names, counts = np.unique(values.astype(str), return_counts=True)
    return str(names[int(np.argmax(counts))])


def action_table(data: Mapping[str, np.ndarray]) -> list[dict[str, object]]:
    action_ids = np.asarray(data["action_id"], dtype=np.int64)
    action_types = np.asarray(data["action_type"]).astype(str)
    action_array = np.asarray(data["action_array"], dtype=np.float32)
    world_delta = np.asarray(data["world_delta"], dtype=np.float32)
    channels = [str(x) for x in np.asarray(data.get("channels", []), dtype=str).tolist()]
    physical_threshold = max(float(np.quantile(world_delta, 0.10)), 1e-8)
    physical = world_delta >= physical_threshold
    detect_thresholds = {}
    for channel in channels:
        key = f"detect_{channel}"
        if key not in data:
            continue
        detect = np.asarray(data[key], dtype=np.float32)
        base = detect[physical]
        detect_thresholds[channel] = max(float(np.quantile(base, 0.10)) if base.size else 1e-8, 1e-8)
    rows: list[dict[str, object]] = []
    for action_id in sorted(np.unique(action_ids).tolist()):
        mask = action_ids == int(action_id)
        arr = action_array[mask][0]
        row = {
            "action_id": int(action_id),
            "action_type": mode_str(action_types[mask]),
            "x": float(arr[0]),
            "y": float(arr[1]),
            "element": int(arr[2]),
            "dx": int(arr[3]),
            "dy": int(arr[4]),
            "mean_world_delta": float(np.mean(world_delta[mask])),
            "std_world_delta": float(np.std(world_delta[mask])),
            "nonnull_rate": float(np.mean(world_delta[mask] > 0)),
            "physical_rate": float(np.mean(physical[mask])),
        }
        for channel in channels:
            key = f"detect_{channel}"
            if key not in data or channel not in detect_thresholds:
                continue
            detect = np.asarray(data[key], dtype=np.float32)
            tau = detect_thresholds[channel]
            row[f"mean_detect_{channel}"] = float(np.mean(detect[mask]))
            row[f"blind_rate_{channel}"] = float(np.mean(np.logical_and(physical[mask], detect[mask] <= tau)))
            row[f"regular_rate_{channel}"] = float(np.mean(np.logical_and(physical[mask], detect[mask] > tau)))
        rows.append(row)
    return rows


def one_hot_balance_score(rows: list[dict[str, object]], probe_ids: set[int]) -> float:
    probe = [row for row in rows if int(row["action_id"]) in probe_ids]
    held = [row for row in rows if int(row["action_id"]) not in probe_ids]
    score = 0.0
    for key in ["action_type", "element", "dx", "dy"]:
        names = sorted({row[key] for row in rows})
        p = np.asarray([np.mean([row[key] == name for row in probe]) for name in names], dtype=np.float32)
        h = np.asarray([np.mean([row[key] == name for row in held]) for name in names], dtype=np.float32)
        score += float(np.sum(np.abs(p - h)))
    for key in ["x", "y", "mean_world_delta", "nonnull_rate", "physical_rate"]:
        all_values = np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        denom = float(np.std(all_values) + 1e-6)
        score += abs(float(np.mean([float(row[key]) for row in probe])) - float(np.mean([float(row[key]) for row in held]))) / denom
    return float(score)


def choose_split(rows: list[dict[str, object]], *, probe_fraction: float, seed: int, n_candidates: int) -> tuple[list[int], list[int], float]:
    action_ids = np.asarray([int(row["action_id"]) for row in rows], dtype=np.int64)
    n_actions = int(action_ids.shape[0])
    n_probe = int(round(float(probe_fraction) * n_actions))
    n_probe = max(1, min(n_actions - 1, n_probe))
    rng = np.random.default_rng(seed)
    best_probe: set[int] | None = None
    best_score = float("inf")

    # Include a deterministic interleaved baseline before random search.
    candidates = [set(action_ids[:: max(1, n_actions // n_probe)][:n_probe].tolist())]
    for _ in range(max(0, int(n_candidates) - 1)):
        candidates.append(set(rng.choice(action_ids, size=n_probe, replace=False).tolist()))

    for probe_ids in candidates:
        if len(probe_ids) != n_probe:
            continue
        score = one_hot_balance_score(rows, probe_ids)
        if score < best_score:
            best_score = score
            best_probe = set(probe_ids)
    if best_probe is None:
        raise RuntimeError("Could not create a valid action split.")
    probe = sorted(best_probe)
    held = sorted(set(action_ids.tolist()) - best_probe)
    return probe, held, best_score


def summarize_split(rows: list[dict[str, object]], probe: set[int]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if key not in {"action_id", "action_type", "split"} and isinstance(value, (int, float, np.integer, np.floating))
    ]
    for split_name, split_rows in [
        ("probe", [row for row in rows if int(row["action_id"]) in probe]),
        ("heldout", [row for row in rows if int(row["action_id"]) not in probe]),
    ]:
        for key in numeric_keys:
            values = np.asarray([float(row[key]) for row in split_rows], dtype=np.float32)
            out.append(
                {
                    "split": split_name,
                    "metric": key,
                    "value": "mean",
                    "count": int(values.size),
                    "stat": float(np.mean(values)),
                }
            )
            out.append(
                {
                    "split": split_name,
                    "metric": key,
                    "value": "std",
                    "count": int(values.size),
                    "stat": float(np.std(values)),
                }
            )
        for key in ["action_type", "element", "dx", "dy"]:
            names = sorted({row[key] for row in rows})
            for name in names:
                out.append(
                    {
                        "split": split_name,
                        "metric": key,
                        "value": str(name),
                        "count": int(len(split_rows)),
                        "stat": float(np.mean([row[key] == name for row in split_rows])),
                    }
                )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data = load_npz(args.data)
    rows = action_table(data)
    probe, held, score = choose_split(
        rows,
        probe_fraction=float(args.probe_fraction),
        seed=int(args.seed),
        n_candidates=int(args.n_candidates),
    )
    out = ensure_dir(args.out)
    (out / "probe_action_ids.txt").write_text(" ".join(str(x) for x in probe) + "\n", encoding="utf-8")
    (out / "heldout_action_ids.txt").write_text(" ".join(str(x) for x in held) + "\n", encoding="utf-8")
    assignments = []
    probe_set = set(probe)
    for row in rows:
        assignments.append({**row, "split": "probe" if int(row["action_id"]) in probe_set else "heldout"})
    write_csv(out / "action_split_assignments.csv", assignments)
    write_csv(out / "action_split_balance.csv", summarize_split(rows, probe_set))
    save_json(
        {
            "data": str(Path(args.data)),
            "probe_action_ids": probe,
            "heldout_action_ids": held,
            "probe_fraction": float(args.probe_fraction),
            "seed": int(args.seed),
            "n_candidates": int(args.n_candidates),
            "balance_score": float(score),
        },
        out / "action_split_summary.json",
    )
    print(f"Wrote balanced action split to {out} with balance_score={score:.6f}")


if __name__ == "__main__":
    main()
