"""PhysPPI deterministic feature representations."""

from openfold3.projects.of3_all_atom.physppi.features._builders import (
    PhysPPIDataError,
    build_interaction_features,
    build_interface_residue_features,
    build_membrane_target,
    build_surface_features,
    build_tm_features,
    iter_mappable_dmap_edges,
    restrict_tm_payload_to_chains,
)
from openfold3.projects.of3_all_atom.physppi.features.surface_position import (
    encode_surface_position,
)

__all__ = [
    "PhysPPIDataError",
    "build_interaction_features",
    "build_interface_residue_features",
    "build_membrane_target",
    "build_surface_features",
    "build_tm_features",
    "encode_surface_position",
    "iter_mappable_dmap_edges",
    "restrict_tm_payload_to_chains",
]
