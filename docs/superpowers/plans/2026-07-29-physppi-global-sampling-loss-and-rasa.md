# PhysPPI Global Sampling, Loss Scaling, and RASA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sample the combined PhysPPI training pool from raw PT weights, scale the complete AFDB training loss by 0.8, and align cropped validation AtomArrays with model outputs.

**Architecture:** Add an opt-in flat datapoint path to the distributed sampler while preserving its existing default. Carry a PhysPPI-specific scalar loss weight in each sample and apply it through one runner helper. Pass the cropped AtomArray to metrics.

**Tech Stack:** Python 3.10, PyTorch, PyTorch Lightning, Pydantic, pytest, YAML.

---

### Task 1: Cropped AtomArray Contract

**Files:**
- Modify: `openfold3/projects/of3_all_atom/physppi/data/adapters/_runtime.py`
- Test: `openfold3/tests/projects/of3_all_atom/physppi/data/test_qbiolip10k_dataset.py`

- [ ] Add a regression test whose full and cropped AtomArrays have different
  identities, lengths, and atom order, then assert that collation preserves
  the exact cropped object and that its length matches `atom_mask` and
  ground-truth positions.
- [ ] Run the focused test and verify that it fails because the full object is returned.
- [ ] Change `build_sample_features()` to expose the exact
  `sample_data["atom_array_cropped"]` object and fail immediately if its atom
  axis disagrees with `atom_mask` or ground-truth positions.
- [ ] Run the focused test and verify that it passes.

### Task 2: Raw Global Datapoint Sampling

**Files:**
- Modify: `openfold3/core/data/framework/stochastic_sampler_dataset.py`
- Modify: `openfold3/core/data/framework/data_module.py`
- Modify: `openfold3/entry_points/validator.py`
- Modify: `openfold3/projects/of3_all_atom/physppi/data/adapters/_runtime.py`
- Test: `openfold3/tests/core/data/framework/test_stochastic_sampler_dataset.py`
- Test: `openfold3/tests/projects/of3_all_atom/physppi/data/test_qbiolip10k_dataset.py`

- [ ] Add tests proving that flat mode follows concatenated datapoint weights and ignores dataset mixture weights.
- [ ] Run the sampler tests and verify that the new constructor option is absent.
- [ ] Add `global_datapoint_sampling: bool = False` to the data-module configuration and pass it into the sampler.
- [ ] Implement one flat `torch.multinomial()` call and map flat indices back to dataset-local pairs.
- [ ] Change the PhysPPI resolver to preserve raw PT values and reject missing, non-finite, or non-positive formal-training weights.
- [ ] Run the focused sampler and adapter tests and verify that they pass.

### Task 3: Complete AFDB Loss Scaling

**Files:**
- Modify: `openfold3/projects/of3_all_atom/config/dataset_configs.py`
- Modify: `openfold3/projects/of3_all_atom/physppi/data/adapters/_runtime.py`
- Modify: `openfold3/projects/of3_all_atom/runner.py`
- Test: `openfold3/tests/projects/of3_all_atom/physppi/data/test_qbiolip10k_dataset.py`
- Test: `openfold3/tests/projects/of3_all_atom/physppi/model/test_dataset_loss_weight.py`

- [ ] Add configuration and feature tests for a positive scalar `loss_weight`.
- [ ] Add runner-helper tests asserting that total loss and every logged loss component are multiplied by the same scalar.
- [ ] Run the tests and verify that the field and helper are absent.
- [ ] Emit `physppi_loss_weight` from the adapter and apply it after complete loss aggregation in both training paths.
- [ ] Run the focused tests and verify that they pass.

### Task 4: Runtime Contract and Documentation

**Files:**
- Modify: `experiments/_runtime/4gpu_train.yaml`
- Modify: `experiments/_runtime/README.md`
- Modify: `openfold3/tests/projects/of3_all_atom/physppi/data/test_runtime_training_config.py`

- [ ] Add failing YAML contract assertions for flat sampling, QBioLiP loss weight 1.0, AFDB loss weight 0.8, and validation loss weight 1.0/default.
- [ ] Enable flat sampling, make legacy dataset mixture weights neutral, and configure independent PhysPPI loss weights.
- [ ] Document the RASA failure, global PT sampling semantics, and AFDB total-loss scaling.
- [ ] Run the runtime contract test and verify that it passes.

### Task 5: Verification and Four-GPU Restart

**Files:**
- Verify all modified files.

- [ ] Run all focused PhysPPI sampler, dataset, runner, and runtime tests.
- [ ] Run `python -m py_compile` for modified Python modules.
- [ ] Run `git diff --check`.
- [ ] Archive the failed RASA logs.
- [ ] Restart `experiments/_runtime/launch_4gpu.sh` on GPUs 0-3.
- [ ] Verify four NCCL ranks, completed sanity validation, and at least one real training step with no traceback.
