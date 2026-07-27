import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.physppi.experiment_config import (
    ExperimentConfig,
    LoadedExperiment,
    SlurmProfileFile,
    load_experiment,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_CONFIG_CASES = (
    (
        Path(
            "configs/experiments/permutation_alignment/"
            "audit_protected_chain_matching.toml"
        ),
        "audit-protected-chain-matching-v1",
        {
            "PHYSPPI_QBIOLIP_ROOT",
            "PHYSPPI_AFDB_ROOT",
            "PHYSPPI_FORMAL_ROOT",
            "PHYSPPI_COMPONENTS_CIF",
            "PHYSPPI_OUTPUT_ROOT",
        },
    ),
    (
        Path(
            "configs/experiments/permutation_alignment/"
            "build_minimal_alignment_arrays.toml"
        ),
        "build-minimal-arrays-v1",
        {
            "PHYSPPI_MSA_CACHE",
            "PHYSPPI_OUTPUT_ROOT",
        },
    ),
    (
        Path("configs/experiments/permutation_alignment/audit_v1_v32_data_errors.toml"),
        "audit-v1-v32-data-errors-v1",
        {
            "PHYSPPI_QBIOLIP_ROOT",
            "PHYSPPI_AFDB_ROOT",
            "PHYSPPI_FORMAL_ROOT",
            "PHYSPPI_OUTPUT_ROOT",
        },
    ),
    (
        Path(
            "configs/experiments/permutation_alignment/"
            "inspect_chain_identity_alignment.toml"
        ),
        "inspect-chain-identity-alignment-v1",
        {
            "PHYSPPI_DEBUG_DATASET",
            "PHYSPPI_DEBUG_SAMPLE_ID",
            "PHYSPPI_QBIOLIP_ROOT",
            "PHYSPPI_AFDB_ROOT",
            "PHYSPPI_FORMAL_ROOT",
            "PHYSPPI_COMPONENTS_CIF",
            "PHYSPPI_OUTPUT_ROOT",
        },
    ),
)
SLURM_PROFILE_CASES = (
    (Path("configs/slurm/cpu_debug.toml"), None),
    (Path("configs/slurm/single_gpu_debug.toml"), 1),
)


def _discover_experiment_configs() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path.relative_to(REPO_ROOT)
            for path in REPO_ROOT.glob("configs/experiments/**/*.toml")
        )
    )


def _discover_slurm_profiles() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path.relative_to(REPO_ROOT)
            for path in REPO_ROOT.glob("configs/slurm/*.toml")
        )
    )


def test_experiment_config_inventory_matches_contract_cases() -> None:
    expected = tuple(sorted(case[0] for case in EXPERIMENT_CONFIG_CASES))

    assert _discover_experiment_configs() == expected


def test_slurm_profile_inventory_matches_contract_cases() -> None:
    expected = tuple(sorted(case[0] for case in SLURM_PROFILE_CASES))

    assert _discover_slurm_profiles() == expected


@pytest.mark.parametrize(
    "config_path",
    _discover_experiment_configs(),
    ids=lambda path: path.name,
)
def test_discovered_experiment_config_has_strict_schema(config_path: Path) -> None:
    raw = tomllib.loads((REPO_ROOT / config_path).read_text())

    config = ExperimentConfig.model_validate(raw)

    assert config.schema_version == 1


@pytest.mark.parametrize(
    "profile_path",
    _discover_slurm_profiles(),
    ids=lambda path: path.name,
)
def test_discovered_slurm_profile_has_strict_schema(profile_path: Path) -> None:
    raw = tomllib.loads((REPO_ROOT / profile_path).read_text())

    profile = SlurmProfileFile.model_validate(raw)

    assert profile.schema_version == 1


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n")
    return path


def _make_real_config_inputs(tmp_path: Path) -> dict[str, str]:
    qbiolip_root = tmp_path / "qbiolip"
    afdb_root = tmp_path / "afdb"
    formal_root = tmp_path / "formal"
    msa_cache = tmp_path / "msa-cache"
    output_root = tmp_path / "output"
    for path in (
        qbiolip_root,
        afdb_root,
        formal_root,
        msa_cache / "colabfold_msas/main",
    ):
        path.mkdir(parents=True)

    components_cif = _touch(tmp_path / "components.cif")
    for relative_path in (
        "reader-index-cache/qbiolip10k-v1.pkl",
        "reader-index-cache/afdb10k-v1.pkl",
        "manifests/qbiolip-audit.json",
        "manifests/afdb-audit.json",
        "manifests/formal-manifest.json",
        "manifests/runtime-failure-ledger.jsonl",
        "audits/v1-v32-full-20k-data-errors/data-error-ids.json",
    ):
        _touch(formal_root / relative_path)
    _touch(msa_cache / "colabfold_msas/mappings/rep_id_to_seq.json")

    return {
        "PHYSPPI_DEBUG_DATASET": "afdb",
        "PHYSPPI_DEBUG_SAMPLE_ID": "AF-fixture",
        "PHYSPPI_QBIOLIP_ROOT": str(qbiolip_root),
        "PHYSPPI_AFDB_ROOT": str(afdb_root),
        "PHYSPPI_FORMAL_ROOT": str(formal_root),
        "PHYSPPI_COMPONENTS_CIF": str(components_cif),
        "PHYSPPI_MSA_CACHE": str(msa_cache),
        "PHYSPPI_OUTPUT_ROOT": str(output_root),
    }


@pytest.mark.parametrize(
    ("config_path", "run_name", "required_environment"),
    EXPERIMENT_CONFIG_CASES,
    ids=[case[0].name for case in EXPERIMENT_CONFIG_CASES],
)
def test_real_permutation_alignment_config_loads(
    tmp_path: Path,
    config_path: Path,
    run_name: str,
    required_environment: set[str],
) -> None:
    environ = _make_real_config_inputs(tmp_path)
    absolute_config_path = REPO_ROOT / config_path

    loaded = load_experiment(
        absolute_config_path,
        repo_root=REPO_ROOT,
        environ=environ,
        validate_module=True,
    )

    assert loaded.config.experiment.module.startswith(
        "experiments.permutation_alignment."
    )
    assert loaded.config.experiment.run_name == run_name
    assert loaded.config.openfold is None
    assert set(loaded.config.environment.required) == required_environment
    assert loaded.profile_path == REPO_ROOT / "configs/slurm/cpu_debug.toml"
    assert loaded.config.artifacts.output_dir.is_relative_to(
        Path(environ["PHYSPPI_OUTPUT_ROOT"])
    )
    assert "/101045/" not in absolute_config_path.read_text()
    assert "/101045/" not in loaded.profile_path.read_text()


@pytest.mark.parametrize(
    ("config_path", "required_environment"),
    [
        (config_path, required_environment)
        for config_path, _, required_environment in EXPERIMENT_CONFIG_CASES
    ],
    ids=[case[0].name for case in EXPERIMENT_CONFIG_CASES],
)
def test_real_permutation_alignment_config_uses_physppi_environment_namespace(
    config_path: Path,
    required_environment: set[str],
) -> None:
    source = (REPO_ROOT / config_path).read_text()
    raw = tomllib.loads(source)
    placeholders = set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*)\}", source))

    assert set(raw["environment"]["required"]) == required_environment
    assert placeholders == required_environment
    assert all(name.startswith("PHYSPPI_") for name in placeholders)


@pytest.mark.parametrize(
    "cache_name",
    ["qbiolip10k-v1.pkl", "afdb10k-v1.pkl"],
)
def test_protected_chain_config_requires_reader_index_caches(
    tmp_path: Path,
    cache_name: str,
) -> None:
    environ = _make_real_config_inputs(tmp_path)
    missing_cache = (
        Path(environ["PHYSPPI_FORMAL_ROOT"]) / "reader-index-cache" / cache_name
    )
    missing_cache.unlink()
    config_path = (
        REPO_ROOT
        / "configs/experiments/permutation_alignment"
        / "audit_protected_chain_matching.toml"
    )

    with pytest.raises(ValueError, match=cache_name):
        load_experiment(
            config_path,
            repo_root=REPO_ROOT,
            environ=environ,
            validate_module=True,
        )


@pytest.mark.parametrize(
    ("profile_path", "gpus_per_node"),
    SLURM_PROFILE_CASES,
    ids=[case[0].name for case in SLURM_PROFILE_CASES],
)
def test_real_slurm_profile_resources(
    profile_path: Path,
    gpus_per_node: int | None,
) -> None:
    absolute_profile_path = REPO_ROOT / profile_path

    profile = SlurmProfileFile.model_validate(
        tomllib.loads(absolute_profile_path.read_text())
    )

    assert profile.schema_version == 1
    assert profile.slurm.nodes == 1
    assert profile.slurm.ntasks_per_node == 1
    assert profile.slurm.cpus_per_task == 8
    assert profile.slurm.memory == "64G"
    assert profile.slurm.time == "04:00:00"
    assert profile.slurm.partition is None
    assert profile.slurm.account is None
    assert profile.slurm.qos is None
    assert profile.slurm.gpus_per_node == gpus_per_node


def _write_minimal_config(root: Path) -> Path:
    (root / "profile.toml").write_text(
        "schema_version = 1\n"
        "[slurm]\n"
        'partition = "profile-partition"\n'
        "nodes = 1\n"
        "cpus_per_task = 4\n"
        'memory = "16G"\n'
        'time = "01:00:00"\n'
    )
    config = root / "experiment.toml"
    config.write_text(
        "schema_version = 1\n"
        "[experiment]\n"
        'name = "alignment"\n'
        'run_name = "audit-v1"\n'
        'module = "tests.fixtures.fake_experiment"\n'
        'args = ["--input", "${DATA_ROOT}/input.json"]\n'
        "[slurm]\n"
        'profile = "profile.toml"\n'
        'job_name = "align-audit"\n'
        "[artifacts]\n"
        'output_dir = "${OUTPUT_ROOT}/audit-v1"\n'
        "[environment]\n"
        'required = ["DATA_ROOT", "OUTPUT_ROOT"]\n'
        "[validation]\n"
        'required_files = ["${DATA_ROOT}/input.json"]\n'
    )
    return config


def _make_inputs(root: Path) -> tuple[Path, dict[str, str]]:
    data = root / "data"
    data.mkdir()
    (data / "input.json").write_text("{}\n")
    return data, {
        "DATA_ROOT": str(data),
        "OUTPUT_ROOT": str(root / "output"),
    }


def test_load_experiment_merges_and_expands(tmp_path: Path) -> None:
    data, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)

    loaded = load_experiment(
        config,
        repo_root=tmp_path,
        environ=environ,
        validate_module=False,
    )

    assert loaded.config.slurm.nodes == 1
    assert loaded.config.slurm.job_name == "align-audit"
    assert loaded.config.experiment.args == (
        "--input",
        str(data / "input.json"),
    )
    assert loaded.config.artifacts.output_dir == tmp_path / "output/audit-v1"
    assert loaded.config.slurm.profile == tmp_path / "profile.toml"
    assert loaded.experiment_path == config
    assert loaded.profile_path == tmp_path / "profile.toml"
    assert loaded.local_path is None
    assert len(loaded) == 4
    assert LoadedExperiment(*loaded) == loaded


def test_unknown_experiment_field_is_rejected(tmp_path: Path) -> None:
    _, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    config.write_text(config.read_text() + "\n[unknown]\nvalue = 1\n")

    with pytest.raises(ValidationError):
        load_experiment(
            config,
            repo_root=tmp_path,
            environ=environ,
            validate_module=False,
        )


def test_missing_required_environment_variable_is_rejected(tmp_path: Path) -> None:
    config = _write_minimal_config(tmp_path)

    with pytest.raises(ValueError, match="DATA_ROOT"):
        load_experiment(
            config,
            repo_root=tmp_path,
            environ={"OUTPUT_ROOT": str(tmp_path / "output")},
            validate_module=False,
        )


def test_local_override_replaces_partition(tmp_path: Path) -> None:
    _, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    local = tmp_path / "local.toml"
    local.write_text('schema_version = 1\n[slurm]\npartition = "local-partition"\n')

    loaded = load_experiment(
        config,
        repo_root=tmp_path,
        local_path=local,
        environ=environ,
        validate_module=False,
    )

    assert loaded.config.slurm.partition == "local-partition"
    assert loaded.local_path == local


def test_local_override_cannot_change_experiment(tmp_path: Path) -> None:
    _, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    local = tmp_path / "local.toml"
    local.write_text('schema_version = 1\n[experiment]\nrun_name = "local-run"\n')

    with pytest.raises(ValidationError):
        load_experiment(
            config,
            repo_root=tmp_path,
            local_path=local,
            environ=environ,
            validate_module=False,
        )


def test_local_list_replaces_experiment_list(tmp_path: Path) -> None:
    _, environ = _make_inputs(tmp_path)
    environ["LOCAL_ROOT"] = str(tmp_path / "local")
    config = _write_minimal_config(tmp_path)
    local = tmp_path / "local.toml"
    local.write_text('schema_version = 1\n[environment]\nrequired = ["LOCAL_ROOT"]\n')

    loaded = load_experiment(
        config,
        repo_root=tmp_path,
        local_path=local,
        environ=environ,
        validate_module=False,
    )

    assert loaded.config.environment.required == ("LOCAL_ROOT",)


def test_merge_rejects_scalar_to_list_type_change(tmp_path: Path) -> None:
    _, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    config.write_text(
        config.read_text().replace(
            'job_name = "align-audit"\n',
            'job_name = "align-audit"\nnodes = [2]\n',
        )
    )

    with pytest.raises(ValueError, match="nodes"):
        load_experiment(
            config,
            repo_root=tmp_path,
            environ=environ,
            validate_module=False,
        )


def test_missing_required_file_is_rejected(tmp_path: Path) -> None:
    config = _write_minimal_config(tmp_path)
    data = tmp_path / "data"
    data.mkdir()

    with pytest.raises(ValueError, match="required file"):
        load_experiment(
            config,
            repo_root=tmp_path,
            environ={
                "DATA_ROOT": str(data),
                "OUTPUT_ROOT": str(tmp_path / "output"),
            },
            validate_module=False,
        )


def test_missing_required_directory_is_rejected(tmp_path: Path) -> None:
    _, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    config.write_text(
        config.read_text()
        + 'required_directories = ["${DATA_ROOT}/missing-directory"]\n'
    )

    with pytest.raises(ValueError, match="required directory"):
        load_experiment(
            config,
            repo_root=tmp_path,
            environ=environ,
            validate_module=False,
        )


def test_undefined_placeholder_is_rejected(tmp_path: Path) -> None:
    _, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    config.write_text(
        config.read_text().replace(
            'args = ["--input", "${DATA_ROOT}/input.json"]',
            'args = ["--input", "${UNDEFINED_ROOT}/input.json"]',
        )
    )

    with pytest.raises(ValueError, match="UNDEFINED_ROOT"):
        load_experiment(
            config,
            repo_root=tmp_path,
            environ=environ,
            validate_module=False,
        )


def test_unbraced_dollar_name_is_not_expanded(tmp_path: Path) -> None:
    _, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    config.write_text(
        config.read_text().replace(
            'args = ["--input", "${DATA_ROOT}/input.json"]',
            'args = ["--literal", "$DATA_ROOT"]',
        )
    )

    loaded = load_experiment(
        config,
        repo_root=tmp_path,
        environ=environ,
        validate_module=False,
    )

    assert loaded.config.experiment.args == ("--literal", "$DATA_ROOT")


def test_relative_runner_yaml_is_resolved_from_repo_root(tmp_path: Path) -> None:
    _, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    runner = tmp_path / "configs/runner.yaml"
    runner.parent.mkdir()
    runner.write_text("trainer: {}\n")
    config.write_text(
        config.read_text() + "\n[openfold]\n" + 'runner_yaml = "configs/runner.yaml"\n'
    )

    loaded = load_experiment(
        config,
        repo_root=tmp_path,
        environ=environ,
        validate_module=False,
    )

    assert loaded.config.openfold is not None
    assert loaded.config.openfold.runner_yaml == runner


def test_missing_runner_yaml_is_rejected(tmp_path: Path) -> None:
    _, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    config.write_text(
        config.read_text() + "\n[openfold]\n" + 'runner_yaml = "configs/missing.yaml"\n'
    )

    with pytest.raises(ValueError, match="runner YAML"):
        load_experiment(
            config,
            repo_root=tmp_path,
            environ=environ,
            validate_module=False,
        )


def test_missing_module_is_rejected_without_executing_it(tmp_path: Path) -> None:
    _, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    config.write_text(
        config.read_text().replace(
            'module = "tests.fixtures.fake_experiment"',
            'module = "package_that_does_not_exist.experiment"',
        )
    )

    with pytest.raises(ValueError, match="module"):
        load_experiment(
            config,
            repo_root=tmp_path,
            environ=environ,
        )


def test_existing_output_directory_is_rejected(tmp_path: Path) -> None:
    _, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    (tmp_path / "output/audit-v1").mkdir(parents=True)

    with pytest.raises(ValueError, match="output"):
        load_experiment(
            config,
            repo_root=tmp_path,
            environ=environ,
            validate_module=False,
        )


def test_filesystem_root_output_is_rejected(tmp_path: Path) -> None:
    data, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    config.write_text(
        config.read_text().replace(
            'output_dir = "${OUTPUT_ROOT}/audit-v1"',
            'output_dir = "/"',
        )
    )
    environ["DATA_ROOT"] = str(data)

    with pytest.raises(ValueError, match="filesystem root"):
        load_experiment(
            config,
            repo_root=tmp_path,
            environ=environ,
            validate_module=False,
        )


def test_relative_input_paths_are_resolved_from_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _, environ = _make_inputs(repo_root)
    config = _write_minimal_config(repo_root)
    config_dir = repo_root / "configs"
    config_dir.mkdir()
    experiment_path = config_dir / "experiment.toml"
    config.rename(experiment_path)
    local_path = config_dir / "local.toml"
    local_path.write_text(
        'schema_version = 1\n[slurm]\npartition = "local-partition"\n'
    )
    outside_repo = tmp_path / "outside"
    outside_repo.mkdir()
    monkeypatch.chdir(outside_repo)

    loaded = load_experiment(
        Path("configs/experiment.toml"),
        repo_root=repo_root,
        local_path=Path("configs/local.toml"),
        environ=environ,
        validate_module=False,
    )

    assert loaded.config.slurm.partition == "local-partition"
    assert loaded.experiment_path == experiment_path
    assert loaded.local_path == local_path


@pytest.mark.parametrize("invalid_version", ["true", "1.0"])
@pytest.mark.parametrize("file_kind", ["experiment", "profile", "local"])
def test_schema_version_rejects_non_integer_one(
    tmp_path: Path,
    invalid_version: str,
    file_kind: str,
) -> None:
    _, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    local = tmp_path / "local.toml"
    local.write_text("schema_version = 1\n")
    target = {
        "experiment": config,
        "profile": tmp_path / "profile.toml",
        "local": local,
    }[file_kind]
    target.write_text(
        target.read_text().replace(
            "schema_version = 1",
            f"schema_version = {invalid_version}",
            1,
        )
    )

    with pytest.raises(ValidationError, match="schema_version"):
        load_experiment(
            config,
            repo_root=tmp_path,
            local_path=local,
            environ=environ,
            validate_module=False,
        )


def test_job_name_rejects_newline(tmp_path: Path) -> None:
    _, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    config.write_text(
        config.read_text().replace(
            'job_name = "align-audit"',
            'job_name = "align\\naudit"',
        )
    )

    with pytest.raises(ValueError, match="control character"):
        load_experiment(
            config,
            repo_root=tmp_path,
            environ=environ,
            validate_module=False,
        )


def test_environment_export_rejects_invalid_identifier(tmp_path: Path) -> None:
    _, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    config.write_text(
        config.read_text() + '\n[environment.export]\n"INVALID-NAME" = "unsafe"\n'
    )

    with pytest.raises(ValidationError, match="INVALID-NAME"):
        load_experiment(
            config,
            repo_root=tmp_path,
            environ=environ,
            validate_module=False,
        )


def test_environment_required_rejects_invalid_identifier(tmp_path: Path) -> None:
    _, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    config.write_text(
        config.read_text().replace(
            'required = ["DATA_ROOT", "OUTPUT_ROOT"]',
            'required = ["INVALID-NAME"]',
        )
    )

    with pytest.raises(ValidationError, match="INVALID-NAME"):
        load_experiment(
            config,
            repo_root=tmp_path,
            environ=environ,
            validate_module=False,
        )


def test_expanded_environment_value_rejects_newline(tmp_path: Path) -> None:
    data, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    environ["DATA_ROOT"] = f"{data}\nINJECTED=value"

    with pytest.raises(ValueError, match="control character"):
        load_experiment(
            config,
            repo_root=tmp_path,
            environ=environ,
            validate_module=False,
        )


@pytest.mark.parametrize("control", ["\v", "\f", "\u2028"])
def test_expanded_environment_value_rejects_all_controls_and_line_separators(
    tmp_path: Path,
    control: str,
) -> None:
    data, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    environ["DATA_ROOT"] = f"{data}{control}INJECTED=value"

    with pytest.raises(ValueError, match="control character"):
        load_experiment(
            config,
            repo_root=tmp_path,
            environ=environ,
            validate_module=False,
        )


def test_module_preflight_does_not_execute_parent_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    config.write_text(
        config.read_text().replace(
            'module = "tests.fixtures.fake_experiment"',
            'module = "preflight_package.experiment"',
        )
    )
    sentinel = tmp_path / "parent-imported"
    package = tmp_path / "preflight_package"
    package.mkdir()
    (package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('imported')\n"
    )
    (package / "experiment.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    load_experiment(
        config,
        repo_root=tmp_path,
        environ=environ,
    )

    assert not sentinel.exists()


@pytest.mark.parametrize("file_kind", ["experiment", "profile", "local"])
def test_malformed_toml_reports_source_path(
    tmp_path: Path,
    file_kind: str,
) -> None:
    _, environ = _make_inputs(tmp_path)
    config = _write_minimal_config(tmp_path)
    local = tmp_path / "local.toml"
    local.write_text("schema_version = 1\n")
    target = {
        "experiment": config,
        "profile": tmp_path / "profile.toml",
        "local": local,
    }[file_kind]
    target.write_text("schema_version = [\n")

    with pytest.raises(ValueError, match=re.escape(str(target))) as exc_info:
        load_experiment(
            config,
            repo_root=tmp_path,
            local_path=local,
            environ=environ,
            validate_module=False,
        )

    assert exc_info.value.__cause__ is not None
