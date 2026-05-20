#!/usr/bin/env python
"""Generate Stage D0 counterfactual audit data.

This script supports `--backend powderworld` (optional upstream package) and
`--backend toy` (smoke test using the existing ToyPowderWorld code path).
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path: sys.path.insert(0, str(_REPO_SRC))
from sae_align.utils.io import ensure_dir, load_json  # type: ignore

def js(o):
    if isinstance(o, dict): return {str(k): js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [js(v) for v in o]
    if isinstance(o, np.ndarray): return js(o.tolist())
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.floating, float)):
        x = float(o); return x if math.isfinite(x) else None
    return o

def make_backend(name, grid_size, seed, device, use_jit):
    if name == "powderworld":
        from sae_align.envs.powderworld_real_adapter import RealPowderworldAdapter
        return RealPowderworldAdapter(world_size=grid_size, seed=seed, device=device, use_jit=use_jit)
    if name == "toy":
        from sae_align.envs.toy_powder import ToyPowderWorld, render_channel, EVENT_NAMES, Action
        toy_event_names = tuple(EVENT_NAMES[:5])
        class ToyAdapter:
            EVENT_NAMES = toy_event_names
            def __init__(self): self.env = ToyPowderWorld(grid_size=grid_size, seed=seed)
            def sample_state(self): return self.env.sample_state().astype(np.uint8)
            def make_action_bank(self, k): return self.env.make_action_bank(k)
            def rollout(self, grid, action, horizon):
                rr = self.env.rollout(grid, action, horizon=horizon)
                return type("Roll", (), {"final_grid": rr.final_grid.astype(np.uint8), "event_vector": rr.event_counts[:5].astype(np.float32)})()
            def noop_rollout(self, grid, horizon):
                rr = self.env.rollout(grid, Action("noop"), horizon=horizon)
                return type("Roll", (), {"final_grid": rr.final_grid.astype(np.uint8), "event_vector": rr.event_counts[:5].astype(np.float32)})()
            def render_channel(self, grid, channel, action=None):
                if channel == "event_response": raise ValueError("event_response is target/diagnostic, not input")
                return render_channel(grid, channel, action=action, event_counts=np.zeros(len(EVENT_NAMES), dtype=np.float32), noise_seed=seed)
        return ToyAdapter()
    raise ValueError(name)

def detect(delta): return float(np.mean(np.abs(delta.astype(np.float32))))
def world_delta(a, b): return float(np.mean(np.asarray(a) != np.asarray(b)))

def summary_stats(values):
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return {"mean": None, "std": None, "q05": None, "q50": None, "q95": None, "positive_rate": None}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q50": float(np.quantile(x, 0.50)),
        "q95": float(np.quantile(x, 0.95)),
        "positive_rate": float(np.mean(x > 0)),
    }

def dense_keep(n_states, k_actions, max_rows, mode, rng):
    n = n_states * k_actions; max_rows = min(max(1, int(max_rows)), n)
    if mode == "random":
        keep = np.zeros(n, dtype=bool); keep[rng.choice(n, size=max_rows, replace=False)] = True
        return keep, np.unique(np.nonzero(keep)[0] // k_actions).astype(np.int32)
    if max_rows < k_actions: raise ValueError("full-states requires max_delta_samples >= k_actions")
    n_dense_states = min(n_states, max_rows // k_actions)
    sids = np.sort(rng.choice(n_states, size=n_dense_states, replace=False)).astype(np.int32)
    keep = np.zeros(n, dtype=bool)
    for sid in sids: keep[int(sid) * k_actions : int(sid) * k_actions + k_actions] = True
    return keep, sids

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/staged0_powderworld.json")
    p.add_argument("--out", required=True)
    p.add_argument("--backend", choices=["powderworld", "toy"], default=None)
    p.add_argument("--n-states", type=int); p.add_argument("--k-actions", type=int)
    p.add_argument("--grid-size", type=int); p.add_argument("--horizon", type=int); p.add_argument("--seed", type=int)
    p.add_argument("--device", default=None); p.add_argument("--use-jit", action="store_true")
    p.add_argument("--channels", nargs="*"); p.add_argument("--max-delta-samples", type=int)
    p.add_argument("--dense-sampling", choices=["random", "full-states"])
    p.add_argument("--min-event-prevalence", type=float, default=None)
    p.add_argument("--max-event-prevalence", type=float, default=None)
    return p.parse_args()

def main():
    a = parse_args(); cfg = load_json(a.config) if Path(a.config).exists() else {}
    backend = a.backend or str(cfg.get("backend", "powderworld"))
    n_states = a.n_states or int(cfg.get("n_states", 128)); k_actions = a.k_actions or int(cfg.get("k_actions", 32))
    grid_size = a.grid_size or int(cfg.get("grid_size", 64)); horizon = a.horizon or int(cfg.get("horizon", 4))
    seed = a.seed if a.seed is not None else int(cfg.get("seed", 0)); device = a.device or str(cfg.get("device", "cpu"))
    use_jit = bool(a.use_jit or cfg.get("use_jit", False)); channels = a.channels or list(cfg.get("channels", ["rgb", "range", "local"]))
    max_rows = a.max_delta_samples if a.max_delta_samples is not None else int(cfg.get("max_delta_samples", n_states * k_actions))
    mode = a.dense_sampling or str(cfg.get("dense_sampling", "full-states"))
    min_event_prevalence = float(a.min_event_prevalence if a.min_event_prevalence is not None else cfg.get("min_event_prevalence", 0.02))
    max_event_prevalence = float(a.max_event_prevalence if a.max_event_prevalence is not None else cfg.get("max_event_prevalence", 0.98))
    env = make_backend(backend, grid_size, seed, device, use_jit); rng = np.random.default_rng(seed + 2027)
    states = [env.sample_state() for _ in range(n_states)]; actions = env.make_action_bank(k_actions); n = n_states * k_actions
    action_dim = int(np.asarray(actions[0].as_array()).reshape(-1).size) if actions else 0
    keep, dense_states = dense_keep(n_states, k_actions, max_rows, mode, rng); dense_idx = np.nonzero(keep)[0].astype(np.int64)
    state_id = np.zeros(n, dtype=np.int32); action_id = np.zeros(n, dtype=np.int32); action_array = np.zeros((n, action_dim), dtype=np.int16); action_type = np.empty(n, dtype=object)
    wdelta = np.zeros(n, dtype=np.float32); evdim = len(getattr(env, "EVENT_NAMES", ("event",))); delta_event = np.zeros((n, evdim), dtype=np.float32)
    detects = {ch: np.zeros(n, dtype=np.float32) for ch in channels}; dense_store = {f"obs0_{ch}": [] for ch in channels}; dense_store.update({f"delta_{ch}": [] for ch in channels}); dense_world = []
    for si, grid in enumerate(states):
        for ai, action in enumerate(actions):
            row = si * k_actions + ai; state_id[row] = si; action_id[row] = ai; action_array[row] = action.as_array(); action_type[row] = getattr(action, "action_type", "action")
            row_seed = int(seed * 1000003 + row)
            if hasattr(env, "set_rollout_seed"):
                env.set_rollout_seed(row_seed)
            noop = env.noop_rollout(grid, horizon=horizon)
            if hasattr(env, "set_rollout_seed"):
                env.set_rollout_seed(row_seed)
            act = env.rollout(grid, action, horizon=horizon); wdelta[row] = world_delta(act.final_grid, noop.final_grid); delta_event[row] = act.event_vector - noop.event_vector
            for ch in channels:
                obs0 = env.render_channel(grid, ch, action=action); obsa = env.render_channel(act.final_grid, ch, action=action); obsn = env.render_channel(noop.final_grid, ch, action=action)
                d = (obsa - obsn).astype(np.float32); detects[ch][row] = detect(d)
                if keep[row]: dense_store[f"obs0_{ch}"].append(obs0.astype(np.float32)); dense_store[f"delta_{ch}"].append(d)
            if keep[row]: dense_world.append((act.final_grid != noop.final_grid).astype(np.float32)[None])
    arrays = {"state_id": state_id, "action_id": action_id, "action_array": action_array, "action_type": action_type.astype(str), "world_delta": wdelta, "delta_event_response": delta_event, "delta_sample_indices": dense_idx}
    for ch, val in detects.items(): arrays[f"detect_{ch}"] = val
    for k, vals in dense_store.items():
        if vals: arrays[k] = np.stack(vals).astype(np.float32)
    if dense_world: arrays["world_delta_dense"] = np.stack(dense_world).astype(np.float32)
    sanity = {
        "world_delta": summary_stats(wdelta),
        "event_prevalence_any": float(np.mean(np.sum(np.abs(delta_event), axis=1) > 0)) if delta_event.size else None,
        "event_magnitude": summary_stats(np.sum(np.abs(delta_event), axis=1)) if delta_event.size else {},
        "detect": {ch: summary_stats(val) for ch, val in detects.items()},
    }
    event_prevalence = sanity["event_prevalence_any"]
    if event_prevalence is not None and not (min_event_prevalence <= float(event_prevalence) <= max_event_prevalence):
        raise RuntimeError(
            "Invalid D0 event target prevalence: "
            f"{event_prevalence:.4f} not in [{min_event_prevalence:.4f}, {max_event_prevalence:.4f}]. "
            "This is a generator/target sanity failure, not a Branch 1/2/3 D0 result."
        )
    meta = {"stage": "D0", "backend": backend, "n_states": n_states, "k_actions": k_actions, "grid_size": grid_size, "horizon": horizon, "seed": seed, "channels": channels, "event_names": list(getattr(env, "EVENT_NAMES", [])), "dense_sampling": mode, "dense_rows": int(dense_idx.size), "dense_state_ids": dense_states.tolist(), "sanity": sanity, "notes": "D0 oracle-positive detector audit dataset; not PSP/Dreamer/RL."}
    arrays["metadata_json"] = np.array(json.dumps(js(meta), sort_keys=True), dtype=object)
    out = Path(a.out); ensure_dir(out.parent); np.savez_compressed(out, **arrays); out.with_suffix(".metadata.json").write_text(json.dumps(js(meta), indent=2, sort_keys=True), encoding="utf-8")
    out.with_suffix(".sanity.json").write_text(json.dumps(js(sanity), indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out} rows={n} dense={dense_idx.size} backend={backend}")
    print(json.dumps(js(sanity), sort_keys=True))
if __name__ == "__main__": main()
