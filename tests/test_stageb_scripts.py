import json
import os
import subprocess
import sys

import numpy as np


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
