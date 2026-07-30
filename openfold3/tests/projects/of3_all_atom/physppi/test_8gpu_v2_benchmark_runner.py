import importlib.util
from pathlib import Path


def _load_runner():
    repo = Path(__file__).resolve().parents[5]
    path = repo / "experiments/_runtime/8gpu_model_v2/benchmark_runner.py"
    spec = importlib.util.spec_from_file_location("physppi_v2_benchmark_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_worker():
    repo = Path(__file__).resolve().parents[5]
    path = repo / "experiments/_runtime/8gpu_model_v2/benchmark_worker.py"
    spec = importlib.util.spec_from_file_location("physppi_v2_benchmark_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fit_time_model_recovers_known_coefficients():
    runner = _load_runner()
    measurements = []
    expected = {
        "t0": 0.25,
        "a0": 0.50,
        "a1": 0.002,
        "a2": 0.00001,
    }
    for n_pad in (256, 384, 512, 640, 768):
        for batch_size in (1, 2, 4):
            seconds = (
                expected["t0"]
                + batch_size
                * (
                    expected["a0"]
                    + expected["a1"] * n_pad
                    + expected["a2"] * n_pad * n_pad
                )
            )
            measurements.append(
                {
                    "n_pad": n_pad,
                    "batch_size": batch_size,
                    "median_step_seconds": seconds,
                    "peak_reserved_gb": 20.0,
                    "status": "ok",
                }
            )

    fit = runner.fit_time_model(measurements)

    assert fit["t0"] == pytest_approx(expected["t0"])
    assert fit["a0"] == pytest_approx(expected["a0"])
    assert fit["a1"] == pytest_approx(expected["a1"])
    assert fit["a2"] == pytest_approx(expected["a2"])
    assert fit["r2"] == pytest_approx(1.0)
    assert fit["full_rank"] is True


def test_fit_b1_length_model_recovers_known_coefficients():
    runner = _load_runner()
    measurements = []
    expected = {
        "b0": 0.75,
        "b1": 0.003,
        "b2": 0.00002,
    }
    for n_pad in (256, 384, 512, 640, 768):
        seconds = expected["b0"] + expected["b1"] * n_pad + expected["b2"] * n_pad * n_pad
        measurements.append(
            {
                "n_pad": n_pad,
                "batch_size": 1,
                "median_step_seconds": seconds,
                "status": "ok",
            }
        )

    fit = runner.fit_b1_length_model(measurements)

    assert fit["b0"] == pytest_approx(expected["b0"])
    assert fit["b1"] == pytest_approx(expected["b1"])
    assert fit["b2"] == pytest_approx(expected["b2"])
    assert fit["r2"] == pytest_approx(1.0)


def test_recommend_batch_sizes_uses_fitted_time_and_memory_limits():
    runner = _load_runner()
    time_fit = {
        "t0": 0.0,
        "a0": 0.0,
        "a1": 0.0,
        "a2": 1.0,
    }
    memory_fit = {
        "m0": 0.0,
        "m1": 0.0,
        "m2": 1.0,
        "m_atom": 0.0,
    }

    recommendations = runner.recommend_batch_sizes(
        n_pads=[256, 384, 512, 640, 768],
        time_fit=time_fit,
        memory_fit=memory_fit,
        target_seconds=768 * 768,
        memory_limit=4 * 384 * 384,
        max_batch_size=8,
    )

    assert [row["n_pad"] for row in recommendations] == [256, 384, 512, 640, 768]
    assert [row["recommended_batch_size"] for row in recommendations] == [8, 4, 2, 1, 1]
    assert recommendations[0]["time_limited_batch_size"] == 8
    assert recommendations[1]["memory_limited_batch_size"] == 4


def test_recommend_batch_sizes_uses_representative_atom_count_for_memory():
    runner = _load_runner()

    recommendations = runner.recommend_batch_sizes(
        n_pads=[256],
        time_fit={"t0": 0.0, "a0": 1.0, "a1": 0.0, "a2": 0.0},
        memory_fit={"m0": 0.0, "m1": 0.0, "m2": 0.0, "m_atom": 0.01},
        target_seconds=10.0,
        memory_limit=25.0,
        max_batch_size=4,
        representative_atom_counts={256: 1000.0},
    )

    assert recommendations[0]["memory_limited_batch_size"] == 2
    assert recommendations[0]["recommended_batch_size"] == 2
    assert recommendations[0]["predicted_peak_reserved_gb"] == pytest_approx(20.0)


def test_build_seq_len_bin_payload_assigns_bucket_batch_and_length_weight():
    runner = _load_runner()
    rows = [
        {
            "dataset": "afdb",
            "case_id": "short",
            "length_total": 120,
            "n_crop": 120,
            "bucket": "001_256",
            "identity_weight": 2.0,
        },
        {
            "dataset": "qbiolip",
            "case_id": "mid",
            "length_total": 500,
            "n_crop": 500,
            "bucket": "385_512",
            "identity_weight": 0.5,
        },
        {
            "dataset": "qbiolip",
            "case_id": "long",
            "length_total": 1200,
            "n_crop": 768,
            "bucket": "641_768",
            "identity_weight": 1.5,
        },
    ]

    payload = runner.build_seq_len_bin_payload(rows)

    assert payload["schema"] == "physppi_seq_len_bin_labels_v1"
    assert payload["bin_boundaries"] == [
        {"bucket": "001_256", "min_n_crop": 1, "max_n_crop": 256, "n_pad": 256},
        {"bucket": "257_384", "min_n_crop": 257, "max_n_crop": 384, "n_pad": 384},
        {"bucket": "385_512", "min_n_crop": 385, "max_n_crop": 512, "n_pad": 512},
        {"bucket": "513_640", "min_n_crop": 513, "max_n_crop": 640, "n_pad": 640},
        {"bucket": "641_768", "min_n_crop": 641, "max_n_crop": 768, "n_pad": 768},
    ]
    assert payload["case_to_label"]["short"]["bucket"] == "001_256"
    assert payload["case_to_label"]["short"]["n_pad"] == 256
    assert payload["case_to_label"]["short"]["local_batch_size"] == 4
    assert payload["case_to_label"]["mid"]["local_batch_size"] == 2
    assert payload["case_to_label"]["long"]["n_crop"] == 768
    assert payload["bucket_to_case_ids"]["001_256"] == ["short"]
    assert payload["bucket_to_case_ids"]["385_512"] == ["mid"]
    assert payload["bucket_to_case_ids"]["641_768"] == ["long"]
    assert payload["case_to_label"]["short"]["length_loss_weight_inverse_batch"] == pytest_approx(
        0.25 / ((0.25 + 0.5 + 1.0) / 3.0)
    )
    assert payload["case_to_label"]["mid"]["length_loss_weight_inverse_batch"] == pytest_approx(
        0.5 / ((0.25 + 0.5 + 1.0) / 3.0)
    )
    assert payload["case_to_label"]["long"]["length_loss_weight_inverse_batch"] == pytest_approx(
        1.0 / ((0.25 + 0.5 + 1.0) / 3.0)
    )


def test_build_seq_len_bin_payload_reports_missing_manifest_ids():
    runner = _load_runner()
    payload = runner.build_seq_len_bin_payload(
        [
            {
                "dataset": "afdb",
                "case_id": "present",
                "length_total": 256,
                "n_crop": 256,
                "bucket": "001_256",
                "identity_weight": 1.0,
            }
        ],
        expected_train_ids={"afdb": ["present", "missing"], "qbiolip": []},
    )

    assert payload["missing_seq_pt_ids"] == {"afdb": ["missing"], "qbiolip": []}
    assert payload["summary"]["num_missing_seq_pt"] == 1


def test_summarize_step_timings_excludes_warmup_and_uses_rank_max():
    runner = _load_runner()
    timing_rows = [
        {"rank": 0, "batch_idx": 0, "elapsed_seconds": 99.0, "peak_reserved_gb": 10.0},
        {"rank": 1, "batch_idx": 0, "elapsed_seconds": 98.0, "peak_reserved_gb": 11.0},
        {"rank": 0, "batch_idx": 1, "elapsed_seconds": 1.0, "peak_reserved_gb": 12.0},
        {"rank": 1, "batch_idx": 1, "elapsed_seconds": 1.5, "peak_reserved_gb": 13.0},
        {"rank": 0, "batch_idx": 2, "elapsed_seconds": 2.0, "peak_reserved_gb": 14.0},
        {"rank": 1, "batch_idx": 2, "elapsed_seconds": 2.5, "peak_reserved_gb": 15.0},
        {"rank": 0, "batch_idx": 3, "elapsed_seconds": 3.0, "peak_reserved_gb": 16.0},
        {"rank": 1, "batch_idx": 3, "elapsed_seconds": 3.5, "peak_reserved_gb": 17.0},
    ]

    summary = runner.summarize_step_timings(
        timing_rows,
        warmup_steps=1,
        measured_steps=3,
    )

    assert summary["measured_step_count"] == 3
    assert summary["median_step_seconds"] == pytest_approx(2.5)
    assert summary["peak_reserved_gb"] == pytest_approx(17.0)


def test_candidate_yaml_skips_checkpoint_loading_by_default(tmp_path):
    import yaml

    runner = _load_runner()
    cases_json = tmp_path / "cases.json"
    cases_json.write_text(
        """
{
  "schema": "test",
  "cases_per_dataset_bucket": 1,
  "buckets": {
    "001_256": {
      "afdb": ["AF-0000000065826667-model_v1"],
      "qbiolip": ["6mmq_1"]
    }
  }
}
""".strip()
        + "\n"
    )
    candidate = runner.Candidate("001_256", 256, 1)

    candidate_yaml = runner._write_candidate_yaml(
        candidate=candidate,
        base_yaml=runner.DEFAULT_BASE_YAML,
        output_dir=tmp_path / "run",
        cases_json=cases_json,
        max_steps=4,
    )
    data = yaml.safe_load(candidate_yaml.read_text())

    assert data["experiment_settings"]["restart_checkpoint_path"] is None
    assert data["experiment_settings"]["ckpt_load_settings"] == {
        "manual_checkpoint_loading": False,
        "init_from_ema_weights": False,
        "restore_lr_scheduler": False,
        "restore_time_step": False,
        "strict_loading": True,
    }
    assert data["pl_trainer_args"]["limit_val_batches"] == 0
    assert data["pl_trainer_args"]["num_sanity_val_steps"] == 0
    assert data["pl_trainer_args"]["val_check_interval"] == 1
    assert data["data_module_args"]["epoch_len"] == 32
    assert (
        data["model_update"]["custom"]["settings"]["gradient_clipping"][
            "per_sample_clipping"
        ]
        is False
    )


def test_worker_command_uses_torchrun_for_eight_gpus(tmp_path):
    runner = _load_runner()
    command = runner.build_worker_command(
        conda_env=Path("/env"),
        candidate_yaml=tmp_path / "candidate.yaml",
        output_dir=tmp_path / "run",
        warmup_steps=1,
        measured_steps=3,
        seed=42,
        nproc_per_node=8,
    )

    assert command[:4] == [
        "/env/bin/torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=8",
    ]
    assert str(runner.V2_DIR / "benchmark_worker.py") in command
    assert "--runner-yaml" in command
    assert "--timing-dir" in command


def test_worker_mean_loss_weight_accepts_rank_local_batch_vector():
    import torch

    worker = _load_worker()

    weight = worker._mean_positive_loss_weight(
        torch.tensor([0.5, 1.5]),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert float(weight.item()) == pytest_approx(1.0)


def test_worker_benchmark_pad_stacks_matching_first_dim_with_different_tail_dim():
    import torch

    worker = _load_worker()

    stacked = worker._pad_and_stack_tensors_for_benchmark(
        [
            torch.ones((4, 4, 5), dtype=torch.bool),
            torch.ones((4, 4, 7), dtype=torch.bool),
        ]
    )

    assert stacked.shape == (2, 4, 4, 7)
    assert stacked.dtype == torch.bool
    assert stacked[0, :, :, :5].all()
    assert not stacked[0, :, :, 5:].any()
    assert stacked[1].all()


def test_worker_benchmark_collator_crops_inactive_interaction_placeholders():
    import torch

    worker = _load_worker()

    batch = worker._benchmark_batch_collator(
        [
            {
                "token_mask": torch.ones(4),
                "physppi_interaction_active": torch.tensor(True),
                "physppi_interaction_type_mask": torch.ones((4, 4, 5), dtype=torch.bool),
            },
            {
                "token_mask": torch.ones(4),
                "physppi_interaction_active": torch.tensor(False),
                "physppi_interaction_type_mask": torch.ones((4, 4, 7), dtype=torch.bool),
            },
        ]
    )

    assert batch["physppi_interaction_type_mask"].shape == (2, 4, 4, 5)
    assert batch["physppi_interaction_active"].tolist() == [True, False]


def pytest_approx(value):
    import pytest

    return pytest.approx(value, rel=1e-6, abs=1e-6)
