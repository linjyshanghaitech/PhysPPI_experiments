from __future__ import annotations

import importlib.machinery
import os
import re
from collections.abc import Mapping
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    StringConstraints,
    field_validator,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


_ENVIRONMENT_PLACEHOLDER = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
_ENVIRONMENT_IDENTIFIER = r"^[A-Za-z_][A-Za-z0-9_]*$"
_PROHIBITED_CONTROL_CHARACTERS = frozenset("\r\n\0")


def _normalize_path(value: object) -> object:
    if isinstance(value, str):
        return Path(value)
    return value


def _normalize_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


PathValue = Annotated[Path, BeforeValidator(_normalize_path)]
StringTuple = Annotated[tuple[str, ...], BeforeValidator(_normalize_tuple)]
PathTuple = Annotated[tuple[PathValue, ...], BeforeValidator(_normalize_tuple)]
EnvironmentName = Annotated[
    str,
    StringConstraints(pattern=_ENVIRONMENT_IDENTIFIER),
]
EnvironmentTuple = Annotated[
    tuple[EnvironmentName, ...],
    BeforeValidator(_normalize_tuple),
]


class SourceSnapshot(NamedTuple):
    path: Path
    content: bytes


class _SourceSnapshots(NamedTuple):
    experiment: SourceSnapshot
    profile: SourceSnapshot
    local: SourceSnapshot | None


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SchemaVersionModel(StrictModel):
    @field_validator("schema_version", mode="before", check_fields=False)
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("schema_version must be the integer 1")
        return value


class ExperimentSection(StrictModel):
    name: str
    run_name: str
    module: str
    args: StringTuple = ()


class OpenFoldSection(StrictModel):
    runner_yaml: PathValue


class SlurmResources(StrictModel):
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    nodes: int | None = Field(default=None, ge=1)
    ntasks_per_node: int | None = Field(default=None, ge=1)
    cpus_per_task: int | None = Field(default=None, ge=1)
    gpus_per_node: int | None = Field(default=None, ge=0)
    memory: str | None = None
    time: str | None = None


class ExperimentSlurmSection(SlurmResources):
    profile: PathValue
    job_name: str


class ArtifactsSection(StrictModel):
    output_dir: PathValue


class EnvironmentSection(StrictModel):
    required: EnvironmentTuple = ()
    export: dict[EnvironmentName, str] = Field(default_factory=dict)


class ValidationSection(StrictModel):
    required_files: PathTuple = ()
    required_directories: PathTuple = ()


class ExperimentConfig(SchemaVersionModel):
    schema_version: Literal[1]
    experiment: ExperimentSection
    openfold: OpenFoldSection | None = None
    slurm: ExperimentSlurmSection
    artifacts: ArtifactsSection
    environment: EnvironmentSection = Field(default_factory=EnvironmentSection)
    validation: ValidationSection = Field(default_factory=ValidationSection)
    _source_snapshots: _SourceSnapshots | None = PrivateAttr(default=None)


class SlurmProfileFile(SchemaVersionModel):
    schema_version: Literal[1]
    slurm: SlurmResources


class LocalOverrideFile(SchemaVersionModel):
    schema_version: Literal[1]
    slurm: SlurmResources = Field(default_factory=SlurmResources)
    environment: EnvironmentSection = Field(default_factory=EnvironmentSection)


class LoadedExperiment(NamedTuple):
    config: ExperimentConfig
    experiment_path: Path
    profile_path: Path
    local_path: Path | None

    def _sources(self) -> _SourceSnapshots:
        sources = self.config._source_snapshots
        if sources is None:
            raise ValueError("experiment source snapshots are missing")
        return sources

    @property
    def experiment_source(self) -> SourceSnapshot:
        return self._sources().experiment

    @property
    def profile_source(self) -> SourceSnapshot:
        return self._sources().profile

    @property
    def local_source(self) -> SourceSnapshot | None:
        return self._sources().local


def _load_toml(path: Path) -> tuple[dict[str, Any], SourceSnapshot]:
    content = path.read_bytes()
    try:
        parsed = tomllib.load(BytesIO(content))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML in {path}: {exc}") from exc
    return parsed, SourceSnapshot(path=path, content=content)


def _same_value_kind(left: object, right: object) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return isinstance(left, Mapping) and isinstance(right, Mapping)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
    return type(left) is type(right)


def _merge_mappings(
    lower: Mapping[str, Any],
    higher: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    merged = deepcopy(dict(lower))
    for key, higher_value in higher.items():
        field_path = f"{prefix}.{key}" if prefix else key
        if key not in merged:
            merged[key] = deepcopy(higher_value)
            continue

        lower_value = merged[key]
        if not _same_value_kind(lower_value, higher_value):
            raise ValueError(
                f"configuration field {field_path!r} changes type from "
                f"{type(lower_value).__name__} to {type(higher_value).__name__}"
            )
        if isinstance(lower_value, Mapping) and isinstance(higher_value, Mapping):
            merged[key] = _merge_mappings(
                lower_value,
                higher_value,
                prefix=field_path,
            )
        else:
            merged[key] = deepcopy(higher_value)
    return merged


def _without_schema_version(model: StrictModel) -> dict[str, Any]:
    return model.model_dump(
        mode="python",
        exclude={"schema_version"},
        exclude_unset=True,
    )


def _expand_string(value: str, environ: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in environ:
            raise ValueError(f"environment variable {name!r} is not defined")
        return environ[name]

    expanded = _ENVIRONMENT_PLACEHOLDER.sub(replace, value)
    if any(character in expanded for character in _PROHIBITED_CONTROL_CHARACTERS):
        raise ValueError("configuration string contains a prohibited control character")
    return expanded


def _expand_environment(value: Any, environ: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _expand_string(value, environ)
    if isinstance(value, Path):
        return Path(_expand_string(str(value), environ))
    if isinstance(value, dict):
        return {key: _expand_environment(item, environ) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_expand_environment(item, environ) for item in value)
    if isinstance(value, list):
        return [_expand_environment(item, environ) for item in value]
    return value


def _resolve_repo_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _resolve_config_paths(config: dict[str, Any], repo_root: Path) -> None:
    config["slurm"]["profile"] = _resolve_repo_path(
        config["slurm"]["profile"],
        repo_root,
    )
    if "openfold" in config and config["openfold"] is not None:
        config["openfold"]["runner_yaml"] = _resolve_repo_path(
            config["openfold"]["runner_yaml"],
            repo_root,
        )
    config["artifacts"]["output_dir"] = _resolve_repo_path(
        config["artifacts"]["output_dir"],
        repo_root,
    )

    validation = config.get("validation", {})
    validation["required_files"] = tuple(
        _resolve_repo_path(path, repo_root)
        for path in validation.get("required_files", ())
    )
    validation["required_directories"] = tuple(
        _resolve_repo_path(path, repo_root)
        for path in validation.get("required_directories", ())
    )


def _validate_required_environment(
    config: Mapping[str, Any],
    environ: Mapping[str, str],
) -> None:
    environment = config.get("environment", {})
    for name in environment.get("required", ()):
        if name not in environ:
            raise ValueError(f"required environment variable {name!r} is not defined")


def _find_module_spec(
    module_name: str,
) -> importlib.machinery.ModuleSpec | None:
    components = module_name.split(".")
    search_locations = None
    full_name = ""
    for index, component in enumerate(components):
        if not component:
            return None
        full_name = f"{full_name}.{component}" if full_name else component
        module_spec = importlib.machinery.PathFinder.find_spec(
            full_name,
            search_locations,
        )
        if module_spec is None:
            return None
        if (
            index < len(components) - 1
            and module_spec.submodule_search_locations is None
        ):
            return None
        search_locations = module_spec.submodule_search_locations
    return module_spec


def _preflight(config: ExperimentConfig, *, validate_module: bool) -> None:
    output_dir = config.artifacts.output_dir
    if output_dir == Path("/"):
        raise ValueError("artifacts output directory cannot be the filesystem root")
    if output_dir.exists():
        raise ValueError(f"artifacts output path already exists: {output_dir}")

    if config.openfold is not None and not config.openfold.runner_yaml.is_file():
        raise ValueError(f"runner YAML is not a file: {config.openfold.runner_yaml}")

    for path in config.validation.required_files:
        if not path.is_file():
            raise ValueError(f"required file is missing or not a file: {path}")
    for path in config.validation.required_directories:
        if not path.is_dir():
            raise ValueError(
                f"required directory is missing or not a directory: {path}"
            )

    if validate_module:
        try:
            module_spec = _find_module_spec(config.experiment.module)
        except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
            module_spec = None
        if module_spec is None:
            raise ValueError(
                f"experiment module cannot be found: {config.experiment.module}"
            )


def load_experiment(
    experiment_path: Path,
    *,
    repo_root: Path,
    local_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    validate_module: bool = True,
) -> LoadedExperiment:
    repo_root = Path(repo_root).resolve()
    experiment_path = _resolve_repo_path(experiment_path, repo_root)
    resolved_local_path = (
        _resolve_repo_path(local_path, repo_root) if local_path is not None else None
    )
    environment = dict(os.environ if environ is None else environ)

    experiment_raw, experiment_source = _load_toml(experiment_path)
    experiment_file = ExperimentConfig.model_validate(experiment_raw)
    expanded_profile = _expand_string(
        str(experiment_file.slurm.profile),
        environment,
    )
    profile_path = _resolve_repo_path(expanded_profile, repo_root)
    profile_raw, profile_source = _load_toml(profile_path)
    profile_file = SlurmProfileFile.model_validate(profile_raw)

    merged = _merge_mappings(
        _without_schema_version(profile_file),
        _without_schema_version(experiment_file),
    )
    merged["schema_version"] = experiment_file.schema_version

    local_source = None
    if resolved_local_path is not None:
        local_raw, local_source = _load_toml(resolved_local_path)
        local_file = LocalOverrideFile.model_validate(local_raw)
        merged = _merge_mappings(
            merged,
            _without_schema_version(local_file),
        )

    _validate_required_environment(merged, environment)
    expanded = _expand_environment(merged, environment)
    _resolve_config_paths(expanded, repo_root)
    config = ExperimentConfig.model_validate(expanded)
    _preflight(config, validate_module=validate_module)
    config._source_snapshots = _SourceSnapshots(
        experiment=experiment_source,
        profile=profile_source,
        local=local_source,
    )

    return LoadedExperiment(
        config=config,
        experiment_path=experiment_path,
        profile_path=profile_path,
        local_path=resolved_local_path,
    )


__all__ = [
    "ArtifactsSection",
    "EnvironmentSection",
    "ExperimentConfig",
    "ExperimentSection",
    "ExperimentSlurmSection",
    "LoadedExperiment",
    "LocalOverrideFile",
    "OpenFoldSection",
    "SlurmProfileFile",
    "SlurmResources",
    "SourceSnapshot",
    "ValidationSection",
    "load_experiment",
]
