import numpy as np

from sae_align.envs.toy_powder import Action, DEFAULT_CHANNELS, ToyPowderWorld, render_channel


def test_rollout_shapes():
    env = ToyPowderWorld(grid_size=16, seed=0)
    grid = env.sample_state()
    action = env.make_action_bank(1)[0]
    out = env.rollout(grid, action, horizon=2)
    assert out.final_grid.shape == grid.shape
    assert out.event_counts.ndim == 1


def test_channels_render():
    env = ToyPowderWorld(grid_size=16, seed=0)
    grid = env.sample_state()
    action = Action("place", x=5, y=5, element=3)
    out = env.rollout(grid, action, horizon=1)
    for ch in DEFAULT_CHANNELS:
        arr = render_channel(out.final_grid, ch, action=action, event_counts=out.event_counts, noise_seed=0)
        assert isinstance(arr, np.ndarray)
        assert arr.size > 0


def test_event_alias_for_backward_compatibility():
    env = ToyPowderWorld(grid_size=16, seed=0)
    grid = env.sample_state()
    action = Action("place", x=5, y=5, element=3)
    out = env.rollout(grid, action, horizon=1)
    a = render_channel(out.final_grid, "event", action=action, event_counts=out.event_counts, noise_seed=0)
    b = render_channel(out.final_grid, "event_response", action=action, event_counts=out.event_counts, noise_seed=0)
    assert np.allclose(a, b)
