import numpy as np

from sae_align.models import StandardizedPCAEncoder, default_stageb_channels, load_transition_encoders, save_transition_encoders


def test_default_stageb_channels_exclude_event_response_by_default():
    channels = ["rgb", "range", "local", "event_response", "semantic", "gray_rgb", "blur_rgb"]
    selected = default_stageb_channels(channels, metadata_defaults=["rgb", "range", "local"])
    assert selected == ["rgb", "range", "local", "gray_rgb", "blur_rgb"]
    assert "event_response" not in selected


def test_default_stageb_channels_can_opt_in_event_response():
    channels = ["rgb", "range", "local", "event_response"]
    selected = default_stageb_channels(channels, include_event_response=True)
    assert selected == ["rgb", "range", "local", "event_response"]


def test_standardized_pca_encoder_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 2, 4, 4)).astype(np.float32)
    encoder = StandardizedPCAEncoder(n_components=5).fit(x)
    emb = encoder.transform(x)
    assert emb.shape == (20, 5)
    assert np.all(np.isfinite(emb))

    path = tmp_path / "encoders.npz"
    save_transition_encoders(str(path), {"rgb": encoder}, metadata={"channels": ["rgb"]})
    loaded, metadata = load_transition_encoders(str(path))
    assert metadata["channels"] == ["rgb"]
    assert np.allclose(loaded["rgb"].transform(x), emb)
