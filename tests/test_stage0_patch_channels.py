import numpy as np
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from sae_align.envs.toy_powder import (
    Action,
    DEFAULT_CHANNELS,
    ToyPowderWorld,
    render_channel,
)

_ANALYZE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_stage0_channels.py"
_SPEC = importlib.util.spec_from_file_location("analyze_stage0_channels", _ANALYZE_PATH)
analyze_stage0_channels = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(analyze_stage0_channels)


def test_patch_channels_render():
    env = ToyPowderWorld(grid_size=16, seed=0)
    grid = env.sample_state()
    action = Action("place", x=3, y=3, element=4)
    roll = env.rollout(grid, action, horizon=2)
    for ch in ["gray_rgb", "blur_rgb", "event_response"]:
        arr = render_channel(roll.final_grid, ch, action=action, event_counts=roll.event_counts, noise_seed=0)
        assert isinstance(arr, np.ndarray)
        assert arr.size > 0


def test_default_channels_include_new_controls():
    assert "event_response" in DEFAULT_CHANNELS
    assert "gray_rgb" in DEFAULT_CHANNELS
    assert "blur_rgb" in DEFAULT_CHANNELS
    assert "event" not in DEFAULT_CHANNELS


def test_event_alias_matches_event_response():
    env = ToyPowderWorld(grid_size=16, seed=0)
    grid = env.sample_state()
    action = Action("place", x=5, y=5, element=3)
    out = env.rollout(grid, action, horizon=1)
    ev = render_channel(out.final_grid, "event", action=action, event_counts=out.event_counts)
    evr = render_channel(out.final_grid, "event_response", action=action, event_counts=out.event_counts)
    np.testing.assert_allclose(ev, evr)


def test_legacy_event_schema_normalizes_and_validates():
    data = {
        "channels": np.array(["rgb", "event"]),
        "world_delta": np.array([0.0, 0.2], dtype=np.float32),
        "detect_rgb": np.array([0.0, 0.1], dtype=np.float32),
        "detect_event": np.array([0.0, 1.0], dtype=np.float32),
        "delta_rgb": np.zeros((2, 3, 4, 4), dtype=np.float32),
        "delta_event": np.zeros((2, 5), dtype=np.float32),
        "action_type": np.array(["place", "erase"]),
    }
    normalized = analyze_stage0_channels.normalize_schema(data)
    assert normalized["channels"].tolist() == ["rgb", "event_response"]
    assert "detect_event_response" in normalized
    summary = analyze_stage0_channels.validate_dense_deltas(normalized, ["rgb", "event_response"])
    assert summary["dense_delta_samples"] == 2


def test_make_stage0_dataset_can_store_static_obs0(tmp_path):
    out = tmp_path / "stage0_static.npz"
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_stage0_dataset.py",
            "--out",
            str(out),
            "--n-states",
            "2",
            "--k-actions",
            "3",
            "--grid-size",
            "8",
            "--horizon",
            "1",
            "--channels",
            "rgb",
            "range",
            "local",
            "event_response",
            "--max-delta-samples",
            "4",
            "--store-static-obs",
            "--max-static-bytes",
            "1000000",
        ],
        check=True,
        env=env,
    )
    with np.load(out, allow_pickle=True) as data:
        dense_idx = data["delta_sample_indices"]
        np.testing.assert_array_equal(data["static_obs_sample_indices"], dense_idx)
        assert data["obs0_rgb"].shape[0] == dense_idx.shape[0] == 4
        assert data["obs0_range"].shape[0] == 4
        assert data["obs0_local"].shape[0] == 4
        assert data["obs0_event_response"].shape == (4, 5)
        assert bool(data["store_static_obs"][0])
        assert int(data["estimated_static_obs_bytes"][0]) > 0


def test_make_stage0_dataset_static_obs_budget_failure(tmp_path):
    out = tmp_path / "stage0_static_too_large.npz"
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/make_stage0_dataset.py",
            "--out",
            str(out),
            "--n-states",
            "1",
            "--k-actions",
            "2",
            "--grid-size",
            "8",
            "--channels",
            "rgb",
            "--max-delta-samples",
            "2",
            "--store-static-obs",
            "--max-static-bytes",
            "1",
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Estimated static obs0 storage is too large" in result.stderr


def test_make_stage0_dataset_full_state_dense_sampling(tmp_path):
    out = tmp_path / "stage0_full_states.npz"
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_stage0_dataset.py",
            "--out",
            str(out),
            "--n-states",
            "5",
            "--k-actions",
            "4",
            "--grid-size",
            "8",
            "--horizon",
            "1",
            "--channels",
            "rgb",
            "range",
            "--max-delta-samples",
            "8",
            "--dense-sampling",
            "full-states",
            "--store-static-obs",
        ],
        check=True,
        env=env,
    )
    with np.load(out, allow_pickle=True) as data:
        sample_indices = data["delta_sample_indices"]
        assert data["dense_sampling"].tolist() == ["full-states"]
        assert sample_indices.shape[0] == 8
        state_ids = data["state_id"][sample_indices]
        action_ids = data["action_id"][sample_indices]
        assert len(np.unique(state_ids)) == 2
        for state_id in np.unique(state_ids):
            np.testing.assert_array_equal(np.sort(action_ids[state_ids == state_id]), np.arange(4))


def test_make_stage0_dataset_full_state_requires_action_bank_budget(tmp_path):
    out = tmp_path / "stage0_bad_full_states.npz"
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/make_stage0_dataset.py",
            "--out",
            str(out),
            "--n-states",
            "2",
            "--k-actions",
            "4",
            "--grid-size",
            "8",
            "--channels",
            "rgb",
            "--max-delta-samples",
            "3",
            "--dense-sampling",
            "full-states",
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requires --max-delta-samples to be at least the action-bank size" in result.stderr
