from collections.abc import Iterator, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal, Self, TypeAlias, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    model_validator,
)

_Key = TypeVar("_Key")
_Value = TypeVar("_Value")
_Item = TypeVar("_Item")


class _ImmutableSequence(tuple[_Item, ...]):
    def __new__(cls, value: Sequence[_Item] = ()) -> Self:
        return super().__new__(cls, value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Sequence) and not isinstance(other, (str, bytes)):
            return tuple(self) == tuple(other)
        return False

    __hash__ = tuple.__hash__

    def __reduce__(self) -> tuple[object, ...]:
        return type(self), (tuple(self),)


class _ImmutableMapping(Mapping[_Key, _Value]):
    __slots__ = ("_data",)

    def __init__(self, value: Mapping[_Key, _Value]) -> None:
        object.__setattr__(self, "_data", MappingProxyType(dict(value)))

    def __getitem__(self, key: _Key) -> _Value:
        return self._data[key]

    def __iter__(self) -> Iterator[_Key]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("immutable mapping")

    def __reduce__(self) -> tuple[object, ...]:
        return type(self), (dict(self._data),)

    def __repr__(self) -> str:
        return repr(dict(self._data))


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _ImmutableMapping(
            {key: _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return _ImmutableSequence(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[_Key, _Value]) -> Mapping[_Key, _Value]:
    frozen = _freeze_value(value)
    if not isinstance(frozen, Mapping):
        raise TypeError("expected mapping")
    return frozen


def _tuple_from_list(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _freeze_sequence(value: tuple[_Item, ...]) -> tuple[_Item, ...]:
    return _ImmutableSequence(value)


FrozenMapping: TypeAlias = Annotated[
    Mapping[_Key, _Value],
    AfterValidator(_freeze_mapping),
    PlainSerializer(dict, return_type=dict[_Key, _Value]),
]
FrozenSequence: TypeAlias = Annotated[
    tuple[_Item, ...],
    BeforeValidator(_tuple_from_list),
    AfterValidator(_freeze_sequence),
]

SCHEMA_VERSION = 2
SchemaVersion = Literal[2]
CropStrategy = Literal["spatial", "spatial_interface"]
RequestedLevel = Literal[0, 1, 2, 3, 4]


class _StrictScreeningModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class DatasetName(StrEnum):
    QBIOLIP = "qbiolip"
    AFDB = "afdb"


class Severity(StrEnum):
    WARNING = "warning"
    HARD_DATA = "hard_data"
    LATEST_PREFLIGHT = "latest_preflight"
    INFRASTRUCTURE = "infrastructure"


class SampleStatus(StrEnum):
    PASS = "pass"
    WARNING = "warning"
    RECOVERED = "recovered"
    EXCLUDED = "excluded"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class Finding(_StrictScreeningModel):
    code: str
    severity: Severity
    message: str
    path: Path | None = None
    exception_type: str | None = None
    evidence_sha256: str | None = None
    evidence_path: Path | None = None
    metrics: FrozenMapping[str, int | float | str | bool] = Field(default_factory=dict)


class HistoricalSeed(_StrictScreeningModel):
    dataset: DatasetName
    sample_id: str
    signatures: FrozenSequence[str]
    sources: FrozenSequence[str]


class AuditView(_StrictScreeningModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    crop_strategy: CropStrategy
    requested_level: RequestedLevel
    contact_cutoff: float = Field(ge=8.0, le=18.0)
    view_index: int = Field(ge=0)
    multichain_probe: bool = False


class SampleRecord(_StrictScreeningModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    ordinal: int
    dataset: DatasetName
    sample_id: str
    status: SampleStatus
    findings: FrozenSequence[Finding]
    audit_views: FrozenSequence[AuditView]
    historical_signatures: FrozenSequence[str]
    elapsed_seconds: float = Field(ge=0)
    source_hashes: FrozenMapping[str, str]
    validator_versions: FrozenMapping[str, str]

    @classmethod
    def _canonical_status(
        cls, findings: Sequence[Finding], historical_signatures: Sequence[str]
    ) -> SampleStatus:
        severities = {finding.severity for finding in findings}
        if Severity.INFRASTRUCTURE in severities:
            return SampleStatus.INFRASTRUCTURE_ERROR
        if severities & {Severity.HARD_DATA, Severity.LATEST_PREFLIGHT}:
            return SampleStatus.EXCLUDED
        if Severity.WARNING in severities:
            return SampleStatus.WARNING
        if historical_signatures:
            return SampleStatus.RECOVERED
        return SampleStatus.PASS

    @classmethod
    def classify(cls, **values) -> Self:
        status = cls._canonical_status(
            values["findings"], values["historical_signatures"]
        )
        return cls(status=status, **values)

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        excluding = {Severity.HARD_DATA, Severity.LATEST_PREFLIGHT}
        severities = {finding.severity for finding in self.findings}
        if self.status is SampleStatus.EXCLUDED and not severities & excluding:
            raise ValueError("excluded record requires an excluding finding")
        if self.status is SampleStatus.WARNING and severities & excluding:
            raise ValueError("warning record cannot contain an excluding finding")
        canonical_status = self._canonical_status(
            self.findings, self.historical_signatures
        )
        if self.status is not canonical_status:
            raise ValueError(f"canonical status {canonical_status.value} required")
        return self


class ScreeningOptions(_StrictScreeningModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    qbiolip_root: Path
    afdb_root: Path
    qbiolip_index_cache: Path
    afdb_index_cache: Path
    qbiolip_canonical_report: Path
    afdb_canonical_report: Path
    history_runs_root: Path
    manifests_dir: Path
    runtime_ledger: Path
    prior_audit: Path
    components_cif: Path
    reference_molecule_directory: Path
    output_root: Path
    workers: int = Field(default=8, ge=1)
    sample_timeout_seconds: int = Field(default=600, ge=1)
    max_attempts: int = Field(default=2, ge=1)
    crop_strategies: tuple[CropStrategy, ...] = (
        "spatial",
        "spatial_interface",
    )
    resume: bool = False


class ScanMetadata(_StrictScreeningModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    scanner_version: str
    git_head: str
    source_roots: FrozenMapping[DatasetName, Path]
    reader_cache_sha256: FrozenMapping[DatasetName, str]
    canonical_report_sha256: FrozenMapping[DatasetName, str]
    dependency_versions: FrozenMapping[str, str]
    worker_count: int
    sample_timeout_seconds: int
    max_attempts: int
    crop_strategies: tuple[CropStrategy, ...]
    cli_options: FrozenMapping[str, str | int | bool | FrozenSequence[str]]

    @model_validator(mode="after")
    def validate_datasets(self) -> Self:
        expected = {DatasetName.QBIOLIP, DatasetName.AFDB}
        if set(self.source_roots) != expected:
            raise ValueError("metadata must contain exactly qbiolip and afdb roots")
        if set(self.reader_cache_sha256) != expected:
            raise ValueError("metadata must contain exactly two reader-cache hashes")
        if set(self.canonical_report_sha256) != expected:
            raise ValueError("metadata must contain exactly two report hashes")
        return self


DatasetSampleId: TypeAlias = Annotated[
    tuple[DatasetName, str], BeforeValidator(_tuple_from_list)
]


class ScanSummary(_StrictScreeningModel):
    schema_version: SchemaVersion = SCHEMA_VERSION
    complete: bool
    source_counts: FrozenMapping[DatasetName, int]
    terminal_record_count: int = Field(ge=0)
    duplicate_terminal_record_ids: FrozenSequence[DatasetSampleId]
    status_counts: FrozenMapping[SampleStatus, int]
    category_counts: FrozenMapping[str, int]
    unresolved_infrastructure_ids: FrozenSequence[str]
    artifact_sha256: FrozenMapping[str, str]

    @model_validator(mode="after")
    def validate_completion(self) -> Self:
        expected = {
            DatasetName.QBIOLIP: 10_000,
            DatasetName.AFDB: 10_000,
        }
        if self.complete and self.source_counts != expected:
            raise ValueError("complete scan must contain exactly 10,000 IDs per source")
        if self.complete and self.terminal_record_count != 20_000:
            raise ValueError(
                "complete scan must contain exactly 20,000 terminal records"
            )
        if self.complete and self.duplicate_terminal_record_ids:
            raise ValueError("complete scan cannot contain duplicate terminal records")
        if self.complete and self.unresolved_infrastructure_ids:
            raise ValueError("complete scan cannot contain infrastructure failures")
        return self
