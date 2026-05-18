import numpy as np

from sae_align.analysis.knn_alignment import (
    action_confound_features,
    cosine_knn_indices,
    label_purity,
    pairwise_overlap_matrix,
    pairwise_stratified_overlap_rows,
    residualize_embeddings,
    excluded_candidate_counts,
    excluded_cosine_knn_indices,
    restricted_candidate_counts,
    restricted_cosine_knn_indices,
    shuffled_assignment_knn_indices,
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


def test_restricted_knn_maps_group_neighbors_to_global_rows():
    x = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [-1.0, 0.0],
            [-0.95, -0.05],
            [-0.9, -0.1],
        ],
        dtype=np.float32,
    )
    labels = np.array(["place", "place", "erase", "erase", "erase"])
    neighbors = restricted_cosine_knn_indices(x, labels, k=2)
    assert neighbors.shape == (5, 2)
    assert neighbors[0].tolist() == [1, -1]
    assert neighbors[1].tolist() == [0, -1]
    assert set(neighbors[2].tolist()) == {3, 4}
    assert np.all(neighbors[neighbors >= 0] < x.shape[0])
    assert label_purity(neighbors, labels) == 1.0
    assert restricted_candidate_counts(labels).tolist() == [1, 1, 2, 2, 2]


def test_shuffled_assignment_knn_keeps_original_row_coordinates():
    x = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ],
        dtype=np.float32,
    )
    neighbors = shuffled_assignment_knn_indices(x, np.array([2, 3, 0, 1]), k=1)
    assert neighbors.shape == (4, 1)
    assert np.all(neighbors[:, 0] != np.arange(4))
    assert np.all(neighbors >= 0)
    assert np.all(neighbors < x.shape[0])


def test_excluded_knn_removes_same_label_candidates():
    x = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [-1.0, 0.0],
            [-0.95, -0.05],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1])
    neighbors = excluded_cosine_knn_indices(x, labels, k=2)
    assert neighbors.shape == (4, 2)
    for i, row in enumerate(neighbors):
        valid = row[row >= 0]
        assert valid.size == 2
        assert np.all(labels[valid] != labels[i])
    assert excluded_candidate_counts(labels).tolist() == [2, 2, 2, 2]


def test_action_residualization_removes_linear_confound_components():
    action_array = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    action_type = np.array(["a", "a", "b", "b", "a", "b"])
    confounds = action_confound_features(action_array, action_type)
    embeddings = np.stack(
        [
            2.0 * confounds[:, 0] - 0.5 * confounds[:, 1],
            confounds[:, 0] + confounds[:, 2],
        ],
        axis=1,
    ).astype(np.float32)
    residual = residualize_embeddings(embeddings, confounds)
    design = np.concatenate([confounds, np.ones((confounds.shape[0], 1), dtype=np.float32)], axis=1)
    assert np.allclose(design.T @ residual, 0.0, atol=1e-5)


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
