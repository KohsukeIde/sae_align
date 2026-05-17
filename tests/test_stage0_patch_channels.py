import numpy as np
import importlib.util
from pathlib import Path

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
