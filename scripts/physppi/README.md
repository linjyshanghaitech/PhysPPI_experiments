# PhysPPI Operations

These scripts are the supported command-line entry points for PhysPPI dataset
screening, semantic training configuration, launch, monitoring, and checkpoint
validation.

Run each command from the repository root with the OpenFold CUDA environment:

```bash
.pixi/envs/openfold3-cuda12/bin/python scripts/physppi/<command>.py --help
```

Reusable screening logic lives in
`openfold3.projects.of3_all_atom.physppi.screening`. Command wrappers may
import that package; production package code must never import `scripts`.

## Experiment launcher

Export the roots required by the selected committed TOML, then prepare a run:

```bash
PHYSPPI_MSA_CACHE=/path/to/msa-cache \
PHYSPPI_OUTPUT_ROOT=/path/to/output \
.pixi/envs/openfold3-cpu/bin/python \
  scripts/physppi/submit_experiment.py \
  configs/experiments/permutation_alignment/build_minimal_alignment_arrays.toml
```

This is the default dry-run behavior: it validates inputs and creates the
auditable run directory and `job.sbatch`, but it does not call `sbatch`.

Apply machine-specific scheduler or environment values only through an
explicit local override. `configs/local/example.toml` is not provided by Git;
first create it from the template in `configs/local/README.md`, then run:

```bash
PHYSPPI_MSA_CACHE=/path/to/msa-cache \
PHYSPPI_OUTPUT_ROOT=/path/to/output \
.pixi/envs/openfold3-cpu/bin/python \
  scripts/physppi/submit_experiment.py \
  configs/experiments/permutation_alignment/build_minimal_alignment_arrays.toml \
  --local-config configs/local/example.toml
```

After creating `configs/local/example.toml` as described in
`configs/local/README.md`, submit only by adding `--submit`:

```bash
PHYSPPI_MSA_CACHE=/path/to/msa-cache \
PHYSPPI_OUTPUT_ROOT=/path/to/output \
.pixi/envs/openfold3-cpu/bin/python \
  scripts/physppi/submit_experiment.py \
  configs/experiments/permutation_alignment/build_minimal_alignment_arrays.toml \
  --local-config configs/local/example.toml \
  --submit
```

Experiment TOML files describe orchestration: the module, argument vector,
resource profile, environment requirements, input validation, and artifact
directory. They reference an OpenFold runner YAML when one is needed; they do
not duplicate model, dataset, optimizer, batch-size, learning-rate, or epoch
settings from that YAML.
