# PhysPPI OpenFold3 Training Data Contract

## Required Feature Keys

### OpenFold3 Token-Level Features
- token_mask: [N_token]
- token_index: [N_token]
- asym_id: [N_token]  
- entity_id: [N_token]
- sym_id: [N_token]
- restype: [N_token]
- msa: [N_token, N_msa]
- profile: [N_token, 22]
- deletion_mean: [N_token]
- token_bonds: [N_token, N_token]
- num_atoms_per_token: [N_token]
- atom_ref_pos: [N_atom, 3]
- atom_ref_mask: [N_atom]
- atom_ref_charge: [N_atom]

### Ground Truth (training only)
- ground_truth.atom_positions: [N_atom, 3]
- ground_truth.atom_resolved_mask: [N_atom]

### PhysPPI Features (optional per feature group)
- physppi_interface_pair_mask_10a: [N_token, N_token]
- physppi_interface_pair_dist_rbf: [N_token, N_token, 16]
- physppi_interface_pair_source: scalar or [batch]
- physppi_surface_local_coords: [N_token, K, 3]
- physppi_surface_features: [N_token, K, F]
- physppi_surface_mask: [N_token, K]
- physppi_membrane_signed_dist: [N_token, 1]
- physppi_membrane_exists: [N_token]
- physppi_membrane_normal: [3]
- physppi_membrane_centroid: [3]

## Token Mapping

PDB residues map to OpenFold3 tokens 1:1 for protein residues. Each PDB chain becomes a contiguous token span. The asym_id distinguishes different chains.

## Converter Decision

Legacy OpenFold v2 feature caches (.pt files from openfold/feature_cache_v3_scan) cannot be directly used as OpenFold3 training batches. They lack:
- token_bonds and num_atoms_per_token  
- atom reference features (atom_ref_pos, atom_ref_mask, atom_ref_charge)
- OpenFold3-compatible msa/profile shapes

A regenerator is needed that:
1. Reads PDB structures
2. Runs OpenFold3 data preprocessing (MSA search or precomputed MSA)
3. Builds OpenFold3 token atom reference features
4. Computes PhysPPI interface/surface/membrane features
5. Writes OpenFold3-ready training batches

Until this regenerator exists or an OpenFold3-ready dataset is supplied, real training is blocked.
