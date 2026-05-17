import numpy as np

from sae_align.analysis.knn_alignment import (
    cosine_knn_indices,
    label_purity,
    pairwise_overlap_matrix,
    pairwise_stratified_overlap_rows,
)
from sae_align.analysis.strata import (
    channel_blind_masks,
    dense_delta_sample_indices,
    effect_bins,
    named_strata,
    pair_channel_strata,
)


def test_cosine_knn_indices_excludes_self_and_tracks_clusters():
    x = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ],
        dtype=np.float32,
    )
    neighbors = cosine_knn_indices(x, k=1)
    assert neighbors.shape == (4, 1)
    assert np.all(neighbors[:, 0] != np.arange(4))
    labels = np.array(["a", "a", "b", "b"])
    assert label_purity(neighbors, labels) == 1.0


def test_pairwise_overlap_matrix():
    a = np.array([[1, 2], [0, 2], [3, 1]])
    b = np.array([[1, 0], [0, 2], [3, 2]])
    names, mat = pairwise_overlap_matrix({"a": a, "b": b}, k=2)
    assert names == ["a", "b"]
    assert mat.shape == (2, 2)
    assert np.allclose(np.diag(mat), 1.0)
    assert 0.0 <= mat[0, 1] <= 1.0


def test_strata_helpers_with_dense_subset():
    data = {
        "world_delta": np.array([0.0, 0.2, 0.4, 0.8], dtype=np.float32),
        "action_type": np.array(["place", "erase", "place", "push"]),
        "delta_sample_indices": np.array([1, 3], dtype=np.int64),
    }
    idx = dense_delta_sample_indices(data, n_dense=2)
    assert idx.tolist() == [1, 3]
    bins, cuts = effect_bins(data["world_delta"], quantiles=(0.5,))
    assert bins.tolist() == [0, 0, 1, 1]
    assert cuts.shape == (1,)
    strata = named_strata(data["world_delta"], data["action_type"])
    assert set(["all", "physical_nonnull", "action_type_place"]).issubset(strata)


def test_channel_blind_masks_and_pairwise_rows():
    data = {
        "world_delta": np.array([0.0, 0.2, 0.4, 0.8], dtype=np.float32),
        "detect_rgb": np.array([0.0, 0.01, 0.5, 0.8], dtype=np.float32),
        "detect_range": np.array([0.0, 0.6, 0.01, 0.7], dtype=np.float32),
    }
    sample_idx = np.array([0, 1, 2, 3], dtype=np.int64)
    physical, blind, regular, thresholds = channel_blind_masks(data, ["rgb", "range"], sample_idx, threshold_quantile=0.34)
    assert "world" in thresholds
    strata = pair_channel_strata(physical, blind, regular, "rgb", "range")
    assert "regular_both" in strata
    neighbors = {
        "rgb": np.array([[1, 2], [0, 2], [3, 1], [2, 1]], dtype=np.int64),
        "range": np.array([[1, 3], [0, 3], [3, 0], [2, 0]], dtype=np.int64),
    }
    rows = pairwise_stratified_overlap_rows(neighbors, {("rgb", "range"): strata}, k=2)
    assert any(row["stratum"] == "blind_either" for row in rows)
