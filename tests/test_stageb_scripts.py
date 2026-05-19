import json
import os
import subprocess
import sys

import numpy as np

from scripts.analyze_stageb6_diagnostics import (
    calibrated_cknna,
    calibrated_cycle_knn,
    calibrated_svcca_mean,
)
from scripts.train_stagec0_prediction import binary_oracle_weights, percentile_rank


def test_stagec0_percentile_rank_is_tie_aware():
    ranks = percentile_rank(np.array([0.0, 1.0, 1.0, 2.0], dtype=np.float32))
    assert np.allclose(ranks, np.array([0.0, 0.5, 0.5, 1.0], dtype=np.float32))


def test_stagec0_binary_oracle_weights_keep_binary_ties():
    weights = binary_oracle_weights(np.array([0, 1, 1, 0], dtype=np.float32), alpha=3.0)
    assert np.allclose(weights[[0, 3]], weights[0])
    assert np.allclose(weights[[1, 2]], weights[1])
    assert weights[1] > weights[0]
    assert np.isclose(float(np.mean(weights)), 1.0)


def test_stageb_scripts_default_to_nonleakage_channels(tmp_path):
    rng = np.random.default_rng(0)
    data_path = tmp_path / "stage0.npz"
    n = 12
    np.savez_compressed(
        data_path,
        world_delta=np.linspace(0.0, 1.0, n, dtype=np.float32),
        action_type=np.array(["place", "erase", "push"] * 4),
        channels=np.array(["rgb", "range", "local", "event_response", "gray_rgb", "blur_rgb"]),
        stageb_default_channels=np.array(["rgb", "range", "local"]),
        diagnostic_only_channels=np.array(["event_response"]),
        delta_sample_indices=np.arange(n, dtype=np.int64),
        action_array=rng.normal(size=(n, 5)).astype(np.float32),
        delta_rgb=rng.normal(size=(n, 3, 4, 4)).astype(np.float32),
        delta_range=rng.normal(size=(n, 4, 4, 4)).astype(np.float32),
        delta_local=rng.normal(size=(n, 4, 3, 3)).astype(np.float32),
        delta_event_response=rng.normal(size=(n, 5)).astype(np.float32),
        delta_gray_rgb=rng.normal(size=(n, 1, 4, 4)).astype(np.float32),
        delta_blur_rgb=rng.normal(size=(n, 3, 4, 4)).astype(np.float32),
        detect_rgb=np.linspace(0.0, 0.3, n, dtype=np.float32),
        detect_range=np.linspace(0.1, 0.4, n, dtype=np.float32),
        detect_local=np.linspace(0.2, 0.5, n, dtype=np.float32),
        detect_event_response=np.linspace(0.0, 1.0, n, dtype=np.float32),
        detect_gray_rgb=np.linspace(0.0, 0.2, n, dtype=np.float32),
        detect_blur_rgb=np.linspace(0.0, 0.25, n, dtype=np.float32),
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    model_dir = tmp_path / "model"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_transition_encoder.py",
            "--data",
            str(data_path),
            "--out",
            str(model_dir),
            "--n-components",
            "3",
            "--max-train-samples",
            "10",
        ],
        check=True,
        env=env,
    )

    metadata = json.loads((model_dir / "transition_encoder_metadata.json").read_text())
    assert metadata["channels"] == ["rgb", "range", "local", "gray_rgb", "blur_rgb"]
    assert "event_response" not in metadata["channels"]

    report_dir = tmp_path / "knn"
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_stratified_knn.py",
            "--data",
            str(data_path),
            "--model",
            str(model_dir / "transition_encoders.npz"),
            "--out",
            str(report_dir),
            "--k",
            "2",
            "--max-points",
            "10",
        ],
        check=True,
        env=env,
    )
    assert (report_dir / "reports" / "stratified_knn.csv").exists()
    assert (report_dir / "reports" / "pairwise_stratified_overlap.csv").exists()
    assert (report_dir / "reports" / "alignment_by_pair_and_stratum.csv").exists()
    assert (report_dir / "reports" / "action_only_overlap.csv").exists()
    assert (report_dir / "reports" / "same_action_type_restricted_knn.csv").exists()
    assert (report_dir / "reports" / "same_action_signature_restricted_knn.csv").exists()
    assert (report_dir / "reports" / "action_residualized_knn_overlap.csv").exists()
    assert (report_dir / "reports" / "action_residualized_pairwise_stratified_overlap.csv").exists()
    assert (report_dir / "reports" / "action_residualization_diagnostics.csv").exists()
    assert (report_dir / "reports" / "stageb_knn_summary.json").exists()
    summary = json.loads((report_dir / "reports" / "stageb_knn_summary.json").read_text())
    assert summary["controls"]["same_action_type"]
    assert not summary["controls"]["same_action_id"]
    assert summary["controls"]["same_action_signature"]
    assert summary["controls"]["action_residualized"]


def test_stageb_scripts_hard_fail_on_event_response_without_allow(tmp_path):
    rng = np.random.default_rng(1)
    data_path = tmp_path / "stage0.npz"
    n = 8
    np.savez_compressed(
        data_path,
        world_delta=np.linspace(0.0, 1.0, n, dtype=np.float32),
        action_type=np.array(["place", "erase"] * 4),
        action_array=rng.normal(size=(n, 5)).astype(np.float32),
        channels=np.array(["event_response"]),
        diagnostic_only_channels=np.array(["event_response"]),
        delta_sample_indices=np.arange(n, dtype=np.int64),
        delta_event_response=rng.normal(size=(n, 5)).astype(np.float32),
        detect_event_response=np.linspace(0.0, 1.0, n, dtype=np.float32),
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_transition_encoder.py",
            "--data",
            str(data_path),
            "--out",
            str(tmp_path / "model"),
            "--channels",
            "event_response",
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Diagnostic-only" in result.stderr


def test_stageb_script_can_train_static_obs0_encoder(tmp_path):
    rng = np.random.default_rng(2)
    data_path = tmp_path / "stage0_static.npz"
    n = 10
    np.savez_compressed(
        data_path,
        world_delta=np.linspace(0.0, 1.0, n, dtype=np.float32),
        action_type=np.array(["place", "erase"] * 5),
        channels=np.array(["rgb", "range", "local"]),
        stageb_default_channels=np.array(["rgb", "range", "local"]),
        static_obs_sample_indices=np.arange(n, dtype=np.int64),
        obs0_rgb=rng.normal(size=(n, 3, 4, 4)).astype(np.float32),
        obs0_range=rng.normal(size=(n, 4, 4, 4)).astype(np.float32),
        obs0_local=rng.normal(size=(n, 4, 3, 3)).astype(np.float32),
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    model_dir = tmp_path / "static_model"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_transition_encoder.py",
            "--data",
            str(data_path),
            "--out",
            str(model_dir),
            "--feature-kind",
            "static",
            "--n-components",
            "3",
            "--max-train-samples",
            "8",
        ],
        check=True,
        env=env,
    )

    metadata = json.loads((model_dir / "transition_encoder_metadata.json").read_text())
    assert metadata["feature_kind"] == "static"
    assert metadata["feature_prefix"] == "obs0"
    assert metadata["channels"] == ["rgb", "range", "local"]
    with np.load(model_dir / "transition_embeddings.npz") as embeddings:
        assert embeddings["embedding_rgb"].shape == (n, 3)
        np.testing.assert_array_equal(embeddings["sample_indices"], np.arange(n, dtype=np.int64))


def test_static_action_effect_compare_script_writes_gain_table(tmp_path):
    rng = np.random.default_rng(3)
    data_path = tmp_path / "stage0_static_compare.npz"
    n = 12
    action_type = np.array(["place", "erase", "push"] * 4)
    np.savez_compressed(
        data_path,
        world_delta=np.linspace(0.0, 1.0, n, dtype=np.float32),
        state_id=np.repeat(np.arange(4, dtype=np.int32), 3),
        action_type=action_type,
        action_id=np.tile(np.arange(3, dtype=np.int32), 4),
        channels=np.array(["rgb", "range", "local"]),
        stageb_default_channels=np.array(["rgb", "range", "local"]),
        delta_sample_indices=np.arange(n, dtype=np.int64),
        static_obs_sample_indices=np.arange(n, dtype=np.int64),
        delta_rgb=rng.normal(size=(n, 3, 4, 4)).astype(np.float32),
        delta_range=rng.normal(size=(n, 4, 4, 4)).astype(np.float32),
        delta_local=rng.normal(size=(n, 4, 3, 3)).astype(np.float32),
        obs0_rgb=rng.normal(size=(n, 3, 4, 4)).astype(np.float32),
        obs0_range=rng.normal(size=(n, 4, 4, 4)).astype(np.float32),
        obs0_local=rng.normal(size=(n, 4, 3, 3)).astype(np.float32),
        detect_rgb=np.linspace(0.0, 0.3, n, dtype=np.float32),
        detect_range=np.linspace(0.1, 0.4, n, dtype=np.float32),
        detect_local=np.linspace(0.2, 0.5, n, dtype=np.float32),
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    action_model_dir = tmp_path / "action_model"
    static_model_dir = tmp_path / "static_model"
    for model_dir, extra in [
        (action_model_dir, []),
        (static_model_dir, ["--feature-kind", "static"]),
    ]:
        subprocess.run(
            [
                sys.executable,
                "scripts/train_transition_encoder.py",
                "--data",
                str(data_path),
                "--out",
                str(model_dir),
                "--n-components",
                "3",
                "--max-train-samples",
                "10",
                *extra,
            ],
            check=True,
            env=env,
        )

    out = tmp_path / "static_compare"
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_static_action_effect_knn.py",
            "--data",
            str(data_path),
            "--action-effect-model",
            str(action_model_dir / "transition_encoders.npz"),
            "--static-model",
            str(static_model_dir / "transition_encoders.npz"),
            "--out",
            str(out),
            "--k",
            "2",
            "--max-points",
            "10",
        ],
        check=True,
        env=env,
    )
    assert (out / "reports" / "static_knn.csv").exists()
    assert (out / "reports" / "action_effect_knn.csv").exists()
    assert (out / "reports" / "delta_minus_static_gain.csv").exists()
    summary = json.loads((out / "reports" / "static_action_effect_summary.json").read_text())
    assert summary["channels"] == ["rgb", "range", "local"]
    assert summary["n_points"] == 10
    assert summary["same_state_static_neighbors_excluded"]


def test_stageb6_literature_metrics_identical_features_are_positive():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(24, 8)).astype(np.float32)
    y = x.copy()

    cycle, cycle_cal = calibrated_cycle_knn(x, y, k=4, repeats=20, seed=0)
    cknna, cknna_cal = calibrated_cknna(x, y, k=4, repeats=20, seed=1)
    svcca, n_corrs, svcca_cal = calibrated_svcca_mean(x, y, components=6, repeats=20, seed=2)

    assert cycle > 0.95
    assert cknna > 0.95
    assert svcca > 0.95
    assert n_corrs >= 1
    assert cycle_cal["calibrated_score"] > 0.0
    assert cknna_cal["calibrated_score"] > 0.0
    assert svcca_cal["calibrated_score"] > 0.0


def test_stageb6_literature_metrics_permutation_null_is_lower():
    rng = np.random.default_rng(43)
    x = rng.normal(size=(32, 10)).astype(np.float32)
    y = x @ rng.normal(size=(10, 7)).astype(np.float32)
    y += 0.01 * rng.normal(size=y.shape).astype(np.float32)
    y_bad = y[rng.permutation(y.shape[0])]

    good_cycle, _ = calibrated_cycle_knn(x, y, k=5, repeats=10, seed=3)
    bad_cycle, _ = calibrated_cycle_knn(x, y_bad, k=5, repeats=10, seed=3)
    good_cka, _ = calibrated_cknna(x, y, k=5, repeats=10, seed=4)
    bad_cka, _ = calibrated_cknna(x, y_bad, k=5, repeats=10, seed=4)

    assert good_cycle > bad_cycle
    assert good_cka > bad_cka


def test_state_signature_knn_script_writes_stageb2_reports(tmp_path):
    rng = np.random.default_rng(4)
    data_path = tmp_path / "stage0_state_signature.npz"
    n_states = 6
    n_actions = 4
    n = n_states * n_actions
    static_perm = np.arange(n - 1, -1, -1, dtype=np.int64)
    state_id = np.repeat(np.arange(n_states, dtype=np.int32), n_actions)
    action_id = np.tile(np.arange(n_actions, dtype=np.int32), n_states)
    action_type = np.array(["place", "erase", "push", "place"] * n_states)
    obs0_rgb = rng.normal(size=(n, 3, 4, 4)).astype(np.float32)
    obs0_range = rng.normal(size=(n, 4, 4, 4)).astype(np.float32)
    obs0_local = rng.normal(size=(n, 4, 3, 3)).astype(np.float32)
    np.savez_compressed(
        data_path,
        world_delta=np.linspace(0.0, 1.0, n, dtype=np.float32),
        state_id=state_id,
        action_id=action_id,
        action_type=action_type,
        action_array=rng.normal(size=(n, 5)).astype(np.float32),
        channels=np.array(["rgb", "range", "local"]),
        stageb_default_channels=np.array(["rgb", "range", "local"]),
        delta_sample_indices=np.arange(n, dtype=np.int64),
        static_obs_sample_indices=static_perm,
        delta_rgb=rng.normal(size=(n, 3, 4, 4)).astype(np.float32),
        delta_range=rng.normal(size=(n, 4, 4, 4)).astype(np.float32),
        delta_local=rng.normal(size=(n, 4, 3, 3)).astype(np.float32),
        obs0_rgb=obs0_rgb[static_perm],
        obs0_range=obs0_range[static_perm],
        obs0_local=obs0_local[static_perm],
        detect_rgb=np.linspace(0.0, 0.3, n, dtype=np.float32),
        detect_range=np.linspace(0.1, 0.4, n, dtype=np.float32),
        detect_local=np.linspace(0.2, 0.5, n, dtype=np.float32),
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    action_model_dir = tmp_path / "action_model"
    static_model_dir = tmp_path / "static_model"
    for model_dir, extra in [
        (action_model_dir, []),
        (static_model_dir, ["--feature-kind", "static"]),
    ]:
        subprocess.run(
            [
                sys.executable,
                "scripts/train_transition_encoder.py",
                "--data",
                str(data_path),
                "--out",
                str(model_dir),
                "--n-components",
                "3",
                "--max-train-samples",
                "20",
                "--train-action-ids",
                "0",
                "1",
                "2",
                *extra,
            ],
            check=True,
            env=env,
        )

    out = tmp_path / "stageb2"
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_state_signature_knn.py",
            "--data",
            str(data_path),
            "--model",
            str(action_model_dir / "transition_encoders.npz"),
            "--static-model",
            str(static_model_dir / "transition_encoders.npz"),
            "--out",
            str(out),
            "--k",
            "2",
            "--max-states",
            "5",
            "--probe-fraction",
            "0.5",
            "--probe-action-ids",
            "0",
            "1",
            "2",
            "--bootstrap-repeats",
            "10",
            "--bootstrap-seed",
            "4",
        ],
        check=True,
        env=env,
    )
    assert (out / "reports" / "state_signature_knn.csv").exists()
    assert (out / "reports" / "primary_state_signature_knn.csv").exists()
    assert (out / "reports" / "heldout_action_signature_knn.csv").exists()
    assert (out / "reports" / "state_strata_fractions.csv").exists()
    assert (out / "reports" / "state_strata_fraction_summary.csv").exists()
    assert (out / "reports" / "state_signature_bootstrap_ci.csv").exists()
    assert (out / "reports" / "state_delta_minus_static_gain.csv").exists()
    assert (out / "reports" / "diagnostic_probe_delta_minus_static_gain.csv").exists()
    summary = json.loads((out / "reports" / "stageb2_state_signature_summary.json").read_text())
    assert summary["n_states"] == 5
    assert len(summary["probe_action_ids"]) == 3
    assert len(summary["heldout_action_ids"]) == 1
    assert summary["encoder_action_split"] == "probe_only"
    assert summary["bootstrap_repeats"] == 10
    assert summary["controls"]["probe_to_heldout_cross"]
    assert summary["controls"]["action_column_shuffled"]

    split_out = tmp_path / "stageb3_split"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_balanced_action_split.py",
            "--data",
            str(data_path),
            "--out",
            str(split_out),
            "--n-candidates",
            "20",
            "--seed",
            "3",
        ],
        check=True,
        env=env,
    )
    assert (split_out / "probe_action_ids.txt").exists()
    assert (split_out / "action_split_balance.csv").exists()

    b3_out = tmp_path / "stageb3"
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_stageb3_gates.py",
            "--data",
            str(data_path),
            "--model",
            str(action_model_dir / "transition_encoders.npz"),
            "--static-model",
            str(static_model_dir / "transition_encoders.npz"),
            "--out",
            str(b3_out),
            "--channels",
            "rgb",
            "range",
            "local",
            "--probe-action-ids",
            "0",
            "1",
            "2",
            "--k",
            "2",
            "--max-states",
            "5",
            "--bootstrap-repeats",
            "5",
            "--normalization-modes",
            "none",
            "probe_global_apply",
            "probe_action_type_apply",
        ],
        check=True,
        env=env,
    )
    assert (b3_out / "reports" / "normalization_sweep_knn.csv").exists()
    assert (b3_out / "reports" / "normalization_sweep_bootstrap_ci.csv").exists()
    assert (b3_out / "reports" / "observability_score_vs_overlap.csv").exists()
    assert (b3_out / "reports" / "gate_summary.csv").exists()
    with open(b3_out / "reports" / "normalization_sweep_knn.csv") as f:
        b3_rows = f.read()
    assert "static_heldout_signature" in b3_rows
    assert "action_column_shuffled_heldout" in b3_rows
    assert "probe_action_type_apply" in b3_rows
    b3_summary = json.loads((b3_out / "reports" / "stageb3_summary.json").read_text())
    assert b3_summary["primary_normalization_modes"] == [
        "none",
        "probe_action_type_apply",
        "probe_global_apply",
    ]

    b4_out = tmp_path / "stageb4"
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_stageb4_reliability.py",
            "--data",
            str(data_path),
            "--model",
            str(action_model_dir / "transition_encoders.npz"),
            "--out",
            str(b4_out),
            "--channels",
            "rgb",
            "range",
            "local",
            "--probe-action-ids",
            "0",
            "1",
            "2",
            "--k",
            "2",
            "--max-states",
            "5",
            "--bootstrap-repeats",
            "5",
            "--normalization-modes",
            "none",
            "probe_global_apply",
            "probe_action_type_apply",
        ],
        check=True,
        env=env,
    )
    assert (b4_out / "reports" / "identity_same_action_reliability.csv").exists()
    assert (b4_out / "reports" / "same_channel_probe_heldout_reliability.csv").exists()
    assert (b4_out / "reports" / "same_channel_vs_shuffled_paired_ci.csv").exists()
    assert (b4_out / "reports" / "feature_tie_diagnostics.csv").exists()
    assert (b4_out / "reports" / "gate_summary_ci.csv").exists()
    b4_summary = json.loads((b4_out / "reports" / "stageb4_summary.json").read_text())
    assert b4_summary["gate_order"][0] == "gate_minus1_identity_same_probe_and_heldout_action"

    b5_out = tmp_path / "stageb5"
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_stageb5_heldout_alignment.py",
            "--data",
            str(data_path),
            "--model",
            str(action_model_dir / "transition_encoders.npz"),
            "--static-model",
            str(static_model_dir / "transition_encoders.npz"),
            "--out",
            str(b5_out),
            "--channels",
            "rgb",
            "range",
            "local",
            "--probe-action-ids",
            "0",
            "1",
            "2",
            "--representations",
            "pca_probe_only",
            "raw_delta",
            "random_projection",
            "--k",
            "2",
            "--max-states",
            "5",
            "--bootstrap-repeats",
            "5",
            "--normalization-modes",
            "none",
            "probe_global_apply",
            "probe_action_type_apply",
        ],
        check=True,
        env=env,
    )
    assert (b5_out / "reports" / "heldout_same_action_cross_channel_alignment.csv").exists()
    assert (b5_out / "reports" / "heldout_action_effect_vs_static.csv").exists()
    assert (b5_out / "reports" / "observability_score_correlation.csv").exists()
    assert (b5_out / "reports" / "feature_tie_diagnostics.csv").exists()
    assert (b5_out / "reports" / "b2_signal_comparison.csv").exists()
    assert (b5_out / "reports" / "gate_summary.csv").exists()
    b5_summary = json.loads((b5_out / "reports" / "stageb5_summary.json").read_text())
    assert b5_summary["gate_order"][0] == "gate0_heldout_same_action_redundancy"

    b6_out = tmp_path / "stageb6"
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_stageb6_diagnostics.py",
            "--data",
            str(data_path),
            "--model",
            str(action_model_dir / "transition_encoders.npz"),
            "--static-model",
            str(static_model_dir / "transition_encoders.npz"),
            "--out",
            str(b6_out),
            "--channels",
            "rgb",
            "range",
            "local",
            "--probe-action-ids",
            "0",
            "1",
            "2",
            "--representations",
            "pca_probe_only",
            "raw_delta",
            "random_projection",
            "--k-values",
            "1",
            "2",
            "--jitter-epsilons",
            "0",
            "1e-6",
            "--jitter-seeds",
            "0",
            "--max-states",
            "5",
            "--normalization-modes",
            "none",
            "probe_action_type_apply",
            "--permutation-repeats",
            "2",
            "--ridge-splits",
            "2",
        ],
        check=True,
        env=env,
    )
    assert (b6_out / "reports" / "b6_knn_sensitivity.csv").exists()
    assert (b6_out / "reports" / "b6_paired_differences.csv").exists()
    assert (b6_out / "reports" / "b6_observability_score_correlation_by_k_jitter.csv").exists()
    assert (b6_out / "reports" / "b6_feature_tie_by_k_jitter.csv").exists()
    assert (b6_out / "reports" / "b6_measurement_sanity.csv").exists()
    assert (b6_out / "reports" / "b6_decision_summary.csv").exists()
    b6_summary = json.loads((b6_out / "reports" / "stageb6_summary.json").read_text())
    assert b6_summary["k_values"] == [1, 2]
    with open(b6_out / "reports" / "b6_measurement_sanity.csv") as f:
        b6_rows = f.read()
    assert "state_flat_linear_cka" in b6_rows
    assert "state_flat_rsa_spearman" in b6_rows
    assert "state_flat_ridge_r2" in b6_rows
    assert "action_conditioned_rsa_spearman_mean" in b6_rows
    assert "calibrated_score" in b6_rows
