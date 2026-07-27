"""Stable public facade for PhysPPI data contracts."""

from openfold3.projects.of3_all_atom.physppi.data.priors import (
    InteractionRecord,
    PriorComplexSample,
    PriorSamplePaths,
)
from openfold3.projects.of3_all_atom.physppi.data.readers import (
    INTERACTION_TYPES,
    AFDB10kReader,
    QBioLiP10kReader,
    QBioLiPSample,
    QBioLiPSamplePaths,
    load_or_create_reader_index,
)


def __getattr__(name: str):
    if name == "PhysPPIPriorComplexDatasetAdapter":
        from openfold3.projects.of3_all_atom.physppi.data.adapters import (
            PhysPPIPriorComplexDatasetAdapter,
        )

        return PhysPPIPriorComplexDatasetAdapter
    raise AttributeError(name)


__all__ = [
    "AFDB10kReader",
    "INTERACTION_TYPES",
    "InteractionRecord",
    "PhysPPIPriorComplexDatasetAdapter",
    "PriorComplexSample",
    "PriorSamplePaths",
    "QBioLiP10kReader",
    "QBioLiPSample",
    "QBioLiPSamplePaths",
    "load_or_create_reader_index",
]
