# PhysPPI Training Test Run — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate identity-based `sampling_weights.pt` into the `OF3DistributedSampler` pipeline and create a minimal single-GPU smoke training run to validate the full PhysPPI data loading + training loop end-to-end.

**Architecture:** The `OF3DistributedSampler._get_random_subset()` already reads `datapoint_probabilities` from `dataset.datapoint_cache` and samples via `torch.multinomial`. We only need to replace the hardcoded uniform `[1.0] * len(sample_ids)` in `create_datapoint_cache_from_reader` with weights loaded from `sampling_weights.pt`. Then create a smoke YAML + launch script targeting GPU 1 for a short val-set training run.

**Tech Stack:** PyTorch, PyTorch Lightning, openfold3 runner, PhysPPI data adapters

## Global Constraints

- All modifications to `_runtime.py` must follow the existing `# PhysPPI modification (Jinyu Lin, ShanghaiTech):` annotation pattern
- All new experiment files go under `experiments/_runtime/`
- GPU 1 only for debugging (`CUDA_VISIBLE_DEVICES=1`); 8 GPUs for formal runs
- Fallback must be graceful: if `sampling_weights.pt` is missing, silently use uniform weights (backward compatible)
- `datapoint_probabilities` must sum to `len(sample_ids)` (mean=1.0) to match the existing `OF3DistributedSampler` behavior

---

### Task 1: Integrate `sampling_weights.pt` into `create_datapoint_cache_from_reader`

**Files:**
- Modify: `openfold3/projects/of3_all_atom/physppi/data/adapters/_runtime.py:1418-1460`

**Interfaces:**
- Consumes: `physppi.manifest` (Path to `split_manifest.json`), `physppi.manifest_dataset_key` (str), `physppi.manifest_partition` (str)
- Produces: `self.datapoint_cache` DataFrame with correct `datapoint_probabilities` column (float64, mean=1.0)

- [ ] **Step 1: Add `sampling_weights.pt` loading after manifest filtering**

In `create_datapoint_cache_from_reader`, after `sample_ids` are filtered by manifest (line ~1445), derive the `sampling_weights.pt` path from the manifest's parent directory:

```python
def create_datapoint_cache_from_reader(self, reader, physppi) -> None:
    sample_ids = list(reader.sample_ids())
    if physppi.manifest is not None:
        # ... existing manifest filtering logic ...
        selected_ids = set(manifest_ids)
        sample_ids = [
            sample_id for sample_id in sample_ids if sample_id in selected_ids
        ]
        # PhysPPI modification (Jinyu Lin, ShanghaiTech):
        #   Load identity-based sampling weights when a manifest is active.
        #   Falls back to uniform weights if sampling_weights.pt is missing.
        weights_path = Path(physppi.manifest).parent / "sampling_weights.pt"
        try:
            weights_data = torch.load(
                weights_path, map_location="cpu", weights_only=False
            )
            raw_weights = weights_data.get("sampling_weights", {})
        except (FileNotFoundError, RuntimeError, KeyError):
            raw_weights = {}
    elif physppi.sample_ids:
        allowed = set(physppi.sample_ids)
        sample_ids = [sample_id for sample_id in sample_ids if sample_id in allowed]
        raw_weights = {}
    else:
        raw_weights = {}
    if physppi.max_samples is not None:
        sample_ids = sample_ids[: int(physppi.max_samples)]

    # Build probabilities column
    if raw_weights:
        probs = np.array(
            [raw_weights.get(sid, 0.0) for sid in sample_ids], dtype=np.float64
        )
        total = probs.sum()
        if total > 0:
            probs = probs * (len(sample_ids) / total)
        else:
            probs = np.ones(len(sample_ids), dtype=np.float64)
    else:
        probs = np.ones(len(sample_ids), dtype=np.float64)

    self.datapoint_cache = pd.DataFrame(
        {
            "pdb_id": sample_ids,
            "preferred_chain_or_interface": [None] * len(sample_ids),
            "datapoint_probabilities": probs,
            "n_clust": [1] * len(sample_ids),
        }
    )
```

- [ ] **Step 2: Add `import numpy as np` and `import torch` at top of `_runtime.py` if missing**

Verify these are already imported (they should be).

- [ ] **Step 3: Verify the integration with a unit test**

Run:
```bash
cd /101045/jinyu/PhysPPI_jinyu/PhysPPI
python3 -c "
import torch
import json
from pathlib import Path

# Verify sampling_weights.pt is loadable
w = torch.load('/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/sampling_weights.pt', map_location='cpu', weights_only=False)
print('Schema:', w['schema'])
print('Seed:', w['seed'])
print('AFDB train IDs:', len(w['datasets']['afdb']['train_ids']))
print('AFDB val IDs:', len(w['datasets']['afdb']['val_ids']))
print('Q-BioLiP train IDs:', len(w['datasets']['qbiolip']['train_ids']))
print('Q-BioLiP val IDs:', len(w['datasets']['qbiolip']['val_ids']))
print('Sampling weights entries:', len(w['sampling_weights']))

# Verify manifest is loadable
m = json.loads(Path('/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/split_manifest.json').read_text())
print('Manifest datasets:', list(m['datasets'].keys()))
print('Manifest schema_version:', m['schema_version'])
"
```

- [ ] **Step 4: Verify syntax of modified _runtime.py**

Run:
```bash
cd /101045/jinyu/PhysPPI_jinyu/PhysPPI
python3 -c "import py_compile; py_compile.compile('openfold3/projects/of3_all_atom/physppi/data/adapters/_runtime.py', doraise=True); print('Syntax OK')"
```

---

### Task 2: Create smoke test runner YAML

**Files:**
- Create: `experiments/_runtime/smoke_test.yaml`

**Interfaces:**
- Produces: A valid openfold3 runner YAML config for a 1-GPU smoke training run
- Uses: val set (2,002 samples), 1 GPU, epoch_len=512, max_steps=32

- [ ] **Step 1: Write the smoke test YAML**

```yaml
experiment_settings:
  mode: train
  output_dir: /101045/jinyu/PhysPPI_jinyu/PhysPPI/experiments/_runtime/smoke_output
  restart_checkpoint_path: last

data_module_args:
  batch_size: 1
  num_workers: 4
  epoch_len: 512
  data_seed: 42
  distinct_samples_per_rank_group: true

logging_config:
  log_lr: true

pl_trainer_args:
  devices: 1
  num_nodes: 1
  precision: bf16-mixed
  max_epochs: 1
  max_steps: 32
  log_every_n_steps: 1

checkpoint_config:
  every_n_epochs: 1
  auto_insert_metric_name: false
  save_last: true
  save_top_k: -1

model_update:
  presets:
    - train
  custom:
    settings:
      training_scope: physppi_priors_only
      train_confidence_only: false
    architecture:
      physppi:
        enabled: true
        optional_prior_conditioner:
          enabled: true
          num_interaction_types: 7
        membrane_geometry:
          enabled: true
          normal_error_bins: 16
          depth_cutoff: 30.0

dataset_configs:
  train:
    qbiolip10k_val_smoke:
      dataset_class: QBioLiP10kDataset
      weight: 1.0
      config:
        debug_mode: true
        template:
          n_templates: 4
          take_top_k: false
        crop:
          token_crop:
            enabled: true
            token_budget: 384
            crop_weights:
              contiguous: 0.0
              spatial: 0.5
              spatial_interface: 0.5
          chain_crop:
            enabled: true
        physppi:
          root: /101045/jinyu/PPI/Prior_out/training/Q-BioLiP-10k-new
          num_surface_neighbors: 128
          manifest: /101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/split_manifest.json
          manifest_dataset_key: qbiolip
          manifest_partition: val_ids
          max_samples: 128
        loss:
          confidence_loss_names: [plddt, pde, pae]
          loss_weights:
            experimentally_resolved: 0.0
            pde: 1.0e-4

dataset_paths:
  qbiolip10k_val_smoke:
    alignments_directory: none
    alignment_db_directory: none
    alignment_array_directory: /shared/openfold3/pdb_training_set/alignment_arrays
    dataset_cache_file: /shared/openfold3/pdb_training_set/dataset_caches/training_cache_with_templates.json
    target_structures_directory: /shared/openfold3/pdb_training_set/preprocessed_pdb_data/standard/structure_files
    target_structure_file_format: npz
    reference_molecule_directory: /shared/openfold3/pdb_training_set/preprocessed_pdb_data/standard/reference_mols
    template_cache_directory: /shared/openfold3/pdb_training_set/templates/train_template_cache
    template_structure_array_directory: /shared/openfold3/pdb_training_set/templates/template_structure_arrays
    template_structures_directory: none
    template_file_format: npz
    ccd_file: /101045/jinyu/PhysPPI/assets/openfold3-physppi-20260724/ccd/components.cif
```

---

### Task 3: Create single-GPU launch script

**Files:**
- Create: `experiments/_runtime/launch_smoke.sh`

**Interfaces:**
- Consumes: `experiments/_runtime/smoke_test.yaml`
- Produces: Running training process on GPU 1, stdout/stderr to `smoke_output/train.log`

- [ ] **Step 1: Write the launch script**

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="$REPO_ROOT/experiments/_runtime/smoke_output"

if [ -d "$OUTPUT_DIR" ]; then
    echo "Removing previous smoke output: $OUTPUT_DIR"
    rm -rf "$OUTPUT_DIR"
fi

export CUDA_VISIBLE_DEVICES=1

echo "=== Smoke Test Configuration ==="
echo "Repo root:   $REPO_ROOT"
echo "Output dir:  $OUTPUT_DIR"
echo "GPU:         $CUDA_VISIBLE_DEVICES"
echo "Command:     run_openfold train --runner-yaml experiments/_runtime/smoke_test.yaml"
echo ""

exec "$REPO_ROOT/.pixi/envs/openfold3-cuda12/bin/run_openfold" train \
    --runner-yaml "$REPO_ROOT/experiments/_runtime/smoke_test.yaml" \
    --seed 42 \
    --data-seed 42 \
    2>&1 | tee "$OUTPUT_DIR/train.log"
```

- [ ] **Step 2: Make script executable**

```bash
chmod +x experiments/_runtime/launch_smoke.sh
```

---

### Task 4: Validate — syntax check and dry-run import

**Files:**
- None created; validates Tasks 1-3

- [ ] **Step 1: Syntax check the modified _runtime.py**

```bash
cd /101045/jinyu/PhysPPI_jinyu/PhysPPI
python3 -c "
import py_compile
py_compile.compile(
    'openfold3/projects/of3_all_atom/physppi/data/adapters/_runtime.py',
    doraise=True,
)
print('_runtime.py: Syntax OK')
"
```

- [ ] **Step 2: Verify YAML parseable**

```bash
cd /101045/jinyu/PhysPPI_jinyu/PhysPPI
python3 -c "
import yaml
with open('experiments/_runtime/smoke_test.yaml') as f:
    cfg = yaml.safe_load(f)
print('YAML parsed OK')
print(f'  devices: {cfg[\"pl_trainer_args\"][\"devices\"]}')
print(f'  epoch_len: {cfg[\"data_module_args\"][\"epoch_len\"]}')
print(f'  manifest: {cfg[\"dataset_configs\"][\"train\"][\"qbiolip10k_val_smoke\"][\"config\"][\"physppi\"][\"manifest\"]}')
print(f'  partition: {cfg[\"dataset_configs\"][\"train\"][\"qbiolip10k_val_smoke\"][\"config\"][\"physppi\"][\"manifest_partition\"]}')
print(f'  max_samples: {cfg[\"dataset_configs\"][\"train\"][\"qbiolip10k_val_smoke\"][\"config\"][\"physppi\"][\"max_samples\"]}')
"
```

- [ ] **Step 3: Verify data paths exist**

```bash
cd /101045/jinyu/PhysPPI_jinyu/PhysPPI
python3 -c "
from pathlib import Path
import torch, json

# Check manifest
m = Path('/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/split_manifest.json')
assert m.is_file(), f'Manifest missing: {m}'
print(f'Manifest OK: {m}')

# Check sampling_weights
w = Path('/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/sampling_weights.pt')
assert w.is_file(), f'Weights missing: {w}'
print(f'Weights OK: {w}')

# Check training data root
root = Path('/101045/jinyu/PPI/Prior_out/training/Q-BioLiP-10k-new')
assert root.is_dir(), f'Training data missing: {root}'
print(f'Training root OK: {root}')

# Check precomputed data
pre = Path('/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/Q-BioLiP-10k-new')
assert pre.is_dir(), f'Precomputed data missing: {pre}'
print(f'Precomputed root OK: {pre}')

# Check CCD
ccd = Path('/101045/jinyu/PhysPPI/assets/openfold3-physppi-20260724/ccd/components.cif')
print(f'CCD OK: {ccd}' if ccd.is_file() else f'CCD MISSING: {ccd}')

# Check pixi env
pixi = Path('/101045/jinyu/PhysPPI_jinyu/PhysPPI/.pixi/envs/openfold3-cuda12/bin/run_openfold')
print(f'Pixi env OK: {pixi}' if pixi.is_file() else f'Pixi env MISSING: {pixi}')
"
```

- [ ] **Step 4: Compute manifest SHA256 (required by launch_semantic_training later)**

```bash
cd /101045/jinyu/PhysPPI_jinyu/PhysPPI
python3 -c "
import hashlib
p = '/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/split_manifest.json'
h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
print(f'Manifest SHA256: {h}')
"
```

---

### Task 5: Run the smoke test

**Files:**
- Logs output to: `experiments/_runtime/smoke_output/train.log`

- [ ] **Step 1: Launch the smoke test**

```bash
cd /101045/jinyu/PhysPPI_jinyu/PhysPPI
bash experiments/_runtime/launch_smoke.sh
```

- [ ] **Step 2: Monitor initial output for errors**

Watch for the first 30-60 seconds. Key things to check:
- Data module initialization: look for `"PhysPPI manifest"` or manifest-related log lines
- Dataset loading: `"Creating datapoint cache"` type messages
- First batch: model forward pass should start within ~2 minutes

- [ ] **Step 3: Post-run — verify training ran**

```bash
cd /101045/jinyu/PhysPPI_jinyu/PhysPPI
python3 -c "
from pathlib import Path
out = Path('experiments/_runtime/smoke_output')
print('Output exists:', out.is_dir())
log = out / 'train.log'
if log.is_file():
    lines = log.read_text().splitlines()
    print(f'Log lines: {len(lines)}')
    for line in lines[-20:]:
        print(line)
"
```
