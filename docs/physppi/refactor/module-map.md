# PhysPPI Refactor Module Map

This map records the production layout at candidate baseline `d2078495`.
Paths are relative to `openfold3/projects/of3_all_atom/physppi`. Public names
listed here are compatibility contracts; private helpers move with their
semantic owner and are not new public API.

| Current path | Public symbols or responsibility | Consumers outside `physppi` | Target path / owner |
| --- | --- | --- | --- |
| `__init__.py` | Stable model-facing exports | `openfold3.projects.of3_all_atom.model`, tests | `__init__.py`; public facade |
| `conditioning.py` | `OptionalPriorOutput`, `OptionalPriorConditioner` | OF3 model, conditioner tests | `model/conditioning.py`; optional-prior fusion |
| `interface_residue_encoder.py` | `InterfaceResidueEncoder` | Conditioner and encoder tests | `model/interface_residue_encoder.py`; per-residue single encoding |
| `feature_injection.py` | `assert_no_esm_features`, `validate_optional_prior_features`, `validate_required_physppi_features`, `inject_physppi_features` | OF3 model and feature/label-boundary tests | `model/feature_injection.py`; batch-to-network injection |
| `interaction_auxiliary.py` | `InteractionValidation`, `InteractionDistanceHead`, `validate_interaction_endpoints`, `interaction_auxiliary_loss` | OF3 model, core loss module, adapter, tests | `model/interaction_auxiliary.py`; interaction head and loss |
| `membrane_geometry.py` | `DirectedPlanePrediction`, `DirectedMembranePlaneEstimator`, `MembraneDiffusionConditioner` | OF3 model, permutation alignment, tests | `model/membrane_geometry.py`; directed membrane plane |
| `pair_constraints.py` | `InteractionPairEncoder` | Conditioner and tests | `model/pair_constraints.py`; typed interaction pairs |
| `surface_encoder.py` | `SurfaceTokenEncoder` | Conditioner and surface tests | `model/surface_encoder.py`; single cross-attention mesh fusion |
| `surface_position.py` | `encode_surface_position` | Surface encoder/tests | `features/surface_position.py`; shared mesh/token position encoding |
| `tm_topology.py` | `TMTopologyEncoder` | Conditioner, membrane tests | `model/tm_topology.py`; TM topology encoding |
| `data/__init__.py` | Current Q-BioLiP reader facade | Core dataset wrappers and tests | `data/__init__.py`; stable data facade |
| `data/qbiolip10k.py` | `QBioLiP10kReader` | Core Q-BioLiP dataset wrapper, tests | `data/readers/qbiolip10k.py`; Q-BioLiP source reader |
| `data/afdb10k.py` | `AFDB10kReader` | Core AFDB dataset wrapper, tests | `data/readers/afdb10k.py`; AFDB source reader |
| `data/reader_index_cache.py` | `load_or_create_reader_index` | Reader modules and cache tests | `data/readers/index_cache.py`; immutable reader index cache |
| `data/prior_complex.py` | `InteractionRecord`, `PriorSamplePaths`, `PriorComplexSample`, prior parsers | Core prior dataset wrapper, readers, adapters, tests | `data/priors/schema.py`; immutable prior schema and parsing |
| `data/optional_augmentation.py` | `OptionalSlot`, `SlotState`, `PriorLevelSelection`, cumulative prior selection/masking | Adapter and augmentation/label tests | `data/priors/augmentation.py`; cumulative optional-slot policy |
| `data/occurrence.py` | `OccurrenceRNG`, `build_occurrence_rng` | Adapter, screening and RNG tests | `data/priors/occurrence.py`; deterministic occurrence RNG |
| `data/chain_views.py` | `RetainedChainView`, chain graph and retained-view sampling | Adapter and chain-view tests | `data/adapters/chain_views.py`; chain restriction policy |
| `data/structure_adapter.py` | `PhysPPIPriorComplexDatasetAdapter`, source-mixture sampling, PDB/MSA/geometry/feature assembly | Core single-dataset wrappers and many integration tests | Decompose into `data/adapters/{structure_io,msa,geometry,chain_selection,feature_assembly,dataset}.py`; orchestration facade is removed after consumer migration |
| `data/qbiolip_features.py` | `PhysPPIDataError`, interface/interaction/surface/TM/membrane builders | Adapter and feature tests | Split into `features/{errors,interface_residue,interaction,surface,tm,membrane}.py`; feature contracts |

At this baseline, importing `data.structure_adapter` in a fresh interpreter
requires `openfold3.core.data.framework` to have completed dataset
registration first; otherwise the framework's eager module discovery imports
the adapter again while it is only partially initialized. The final public
facade should eliminate this ordering dependency.

## Dependency boundary

The intended production direction is:

```text
data.readers -> data.priors
data.adapters -> data.readers + data.priors + features
screening -> data.adapters
model -> feature contracts
OpenFold integration -> public physppi data/model facades
scripts -> public physppi interfaces
```

Production code must not import `openfold3.tests` or `scripts`. The two large
decomposition targets are `data/structure_adapter.py` and
`data/qbiolip_features.py`; their behavior is characterized before code moves.
