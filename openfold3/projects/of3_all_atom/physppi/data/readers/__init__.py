"""Canonical PhysPPI source readers."""

from openfold3.projects.of3_all_atom.physppi.data.readers.afdb10k import (
    AFDB10kReader,
)
from openfold3.projects.of3_all_atom.physppi.data.readers.index_cache import (
    load_or_create_reader_index,
)
from openfold3.projects.of3_all_atom.physppi.data.readers.qbiolip10k import (
    INTERACTION_TYPES,
    InteractionRecord,
    QBioLiP10kReader,
    QBioLiPSample,
    QBioLiPSamplePaths,
)

__all__ = [
    "AFDB10kReader",
    "INTERACTION_TYPES",
    "InteractionRecord",
    "QBioLiP10kReader",
    "QBioLiPSample",
    "QBioLiPSamplePaths",
    "load_or_create_reader_index",
]
