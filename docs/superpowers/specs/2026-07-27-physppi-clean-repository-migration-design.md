# PhysPPI Clean Repository Migration Design

## Objective

Create a new Git repository at
`/101045/jinyu/PhysPPI_jinyu/PhysPPI` from the clean source tree at
`/101045/jinyu/PhysPPI_jinyu/qinwc/openfold-3`.

The new repository will:

- start with new Git history on `main`;
- retain the complete source, tests, environment definitions, and necessary
  documentation required to reproduce the current implementation;
- exclude generated output, caches, checkpoints, runtime state, and historical
  debugging material;
- preserve runtime behavior during the import;
- record source commit `4c1c97d13` as provenance.

No remote will be configured during this migration.

## Source State

The source repository is clean at commit:

```text
4c1c97d13 docs(physppi): refresh canonical project handoff
```

It is ahead of its upstream remote by nine documentation commits. The import
uses the current `HEAD`, not `origin/main`, because the current root
documentation and repository contract are part of the selected baseline.

Only files tracked by this commit are eligible for import. Ignored and
untracked filesystem content is never used as an import source.

## Import Method

Use a Git allowlist export from the source commit. This is preferred over a
filesystem copy because it cannot include local caches or untracked files, and
preferred over exporting everything and deleting a blacklist because the final
repository boundary is explicit.

The import must not copy the source `.git` directory or any source remote
configuration.

## Included Content

Import the following paths when they exist in the source commit:

### Runtime and tests

- `openfold3/`
- `scripts/`
- `examples/`

The complete `openfold3/tests/` tree and its test data remain included. Test
fixtures are source-controlled reproducibility inputs, not generated output.

### Build and environment definitions

- `pyproject.toml`
- `pixi.toml`
- `pixi.lock`
- `docker/`
- `environments/`
- `deepspeed_configs/`
- `devtools/`

### Repository metadata

- `.dockerignore`
- `.gitattributes`
- `.git-blame-ignore-revs`
- `.github/`
- `.readthedocs.yaml`
- `CITATION.cff`
- `CONTRIBUTING`
- `LICENSE`
- `README.md`

The imported `.gitignore` will be hardened for the new repository as described
below.

### Documentation

Keep the general OpenFold3 documentation:

- `docs/Makefile`
- `docs/README.md`
- `docs/environment.yml`
- `docs/imgs/`
- `docs/source/`

Keep only these PhysPPI documents:

- `docs/physppi/baseline_inference.md`
- `docs/physppi/full_msa_source_and_file_structure.md`
- `docs/physppi/of3_training_data_contract.md`
- `docs/physppi/qbiolip_manifest_contract.md`
- `docs/physppi/refactor/module-map.md`
- `docs/physppi/verification.md`

These files describe stable interfaces or current module structure. Historical
handoffs, run reports, reviews, and recovery notes are excluded.

## Excluded Content

Do not import:

- `.git/`;
- `.pixi/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`;
- `output/`, `outputs/`, `logs/`, `runs/`, `remote/`, `scratch/`, `work/`;
- checkpoints and model artifacts such as `*.ckpt`, `*.pt`, and `*.pth`;
- `state/` and `.superpowers/`;
- `CURRENT_STATE.md`, `HANDOFF.md`, `INDEX.md`, and `RUNBOOK.md`;
- the source repository's `docs/superpowers/` history;
- dated PhysPPI status reports, reviews, handoffs, recovery documents, meeting
  reports, and historical training records;
- root `assets/`, which contains publication and promotional media not required
  by the build, tests, or root README.

The source repository currently tracks approximately 71 MB below `output/`.
Those files remain excluded even though Git tracks them in the source.

The migration design stored in the target repository at
`docs/superpowers/specs/2026-07-27-physppi-clean-repository-migration-design.md`
is the sole exception to the source-document exclusion. It documents the new
repository's provenance and was not imported from the old repository.

## Ignore Policy

The new `.gitignore` must retain the useful upstream rules and additionally
exclude:

```gitignore
# Local environments and caches
.pixi/
.venv/
.pytest_cache/
.ruff_cache/
**/__pycache__/

# Runtime data and generated artifacts
output/
outputs/
logs/
runs/
remote/
scratch/
state/
work/

# Model artifacts and local configuration
*.ckpt
*.pt
*.pth
*.log
.env
.env.*
```

Tracked test fixtures such as `.npz`, `.cif`, and alignment files must not be
globally ignored.

## Permutation Alignment Review

The migration does not delete chain-level permutation alignment.

The proposed deletion was based on a valid but insufficient observation:
prediction tokens and ground-truth tokens originate from the same processed
structure, so a source-order identity correspondence exists. That observation
does not imply that identity correspondence is the correct geometric
assignment for the model output.

For identical chains, PDB chain names are arbitrary labels. A mini-rollout can
place the coordinates associated with one identical-chain token block at the
physical site occupied by another identical chain. `token_index` identifies
the token block; it does not constrain which symmetry-equivalent physical site
the generated coordinates occupy.

This is explicitly part of AlphaFold3 Supplementary Methods 4.2: the
mini-rollout prediction is used to permute symmetric ground-truth chains and
ligands before confidence and diffusion supervision. It is not merely an
unchanged AlphaFold2 artifact.

The current source has already replaced the fragile legacy greedy assignment
on the training route with protected matching:

- sequence and molecule-type compatibility are hard constraints;
- deterministic Hungarian assignment replaces unbounded or unsafe matching;
- a candidate permutation is accepted only when it improves RMSD over the
  identity baseline;
- typed matcher failures use a sequence-compatible identity fallback;
- configured loss weights are preserved;
- Kabsch alignment and within-component atom symmetry resolution remain.

The correct baseline for later work is therefore:

1. retain protected chain matching;
2. retain Kabsch coordinate alignment;
3. retain residue and ligand atom-level symmetry resolution;
4. investigate `mol_*` label construction independently where evidence shows
   it remains faulty;
5. add GT `asym_id`, `entity_id`, or `sym_id` only for a defined consumer or
   diagnostic contract, not as a replacement for symmetry-aware chain
   matching.

Any later alignment simplification requires a failing regression fixture and a
comparison against the current protected identity baseline.

## Behavior Preservation

The initial import does not rewrite machine-local defaults or operational
paths embedded in existing source and documentation. Although those paths
reduce portability outside the current environment, changing them during the
repository import would mix behavior changes with provenance establishment.

Path portability, README generalization, and runtime configuration cleanup
will be handled as subsequent, independently tested changes in the new
repository.

## Verification

Before the import commit is considered complete:

1. Confirm the source repository is still clean at `4c1c97d13`.
2. Confirm the target has no Git remote and no source history.
3. Confirm all target files are within the approved allowlist.
4. Confirm excluded artifact and state paths are absent.
5. Confirm no tracked file exceeds GitHub's 100 MB per-file limit.
6. Confirm no checkpoint or environment-secret file is tracked.
7. Confirm the `openfold3` package imports from the target tree.
8. Run focused permutation and protected-chain matching tests.
9. Run a representative repository test subset that does not require external
   datasets, checkpoints, GPUs, or network services.
10. Confirm `git status --short --branch` is clean after the import commit.

If a verification failure reveals a source behavior defect rather than an
import defect, record it without silently changing runtime behavior in the
migration commit.
