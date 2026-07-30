# PhysPPI Global Sampling, Loss Scaling, and RASA Design

## Goal

Make the four-GPU PhysPPI run obey three independent contracts:

1. Sample every training case from one combined QBioLiP+AFDB pool using only
   its raw value from `sampling_weights.pt`.
2. Multiply the complete AFDB training loss by `0.8` after all component
   losses are summed; use `1.0` for QBioLiP and do not scale validation.
3. Give validation metrics the cropped `AtomArray` that matches the model
   atom axis when the 768-token crop is enabled.

## Sampling Architecture

`DataModuleArgs.global_datapoint_sampling` enables a flat sampling path in
`OF3DistributedSampler`. The sampler concatenates each training dataset's
`datapoint_probabilities` in dataset order, invokes `torch.multinomial()` once
on that vector, and maps each flat result back to `(dataset_idx,
datapoint_idx)`.

PhysPPI adapters load raw positive finite values from the explicitly
configured `sampling_weights.pt`. They do not normalize each dataset
separately. Validation partitions never load this file and continue to be
enumerated by their validation data loaders.

The legacy hierarchical dataset-then-datapoint path remains the default for
other experiments. The runtime configuration explicitly opts into flat
sampling.

## Loss Architecture

`PriorComplexSettings.loss_weight` is independent from sampling. Training
configuration sets it to `1.0` for QBioLiP and `0.8` for AFDB; validation uses
the default `1.0`.

Each PhysPPI sample carries a scalar `physppi_loss_weight`. The runner applies
that scalar to the total loss and logged loss breakdown after
`OpenFold3Loss` has combined confidence, diffusion, distogram, pDockQ,
membrane, and interaction losses. The automatic-optimization and manual
per-sample-clipping paths call the same helper.

## Validation AtomArray Contract

PhysPPI samples expose the exact `atom_array_cropped` object used to featurize
the model `atom_mask` as batch `atom_array`. This preserves the same crop mask
and atom order; the implementation must never approximate the result by
slicing the first `N` atoms from the full array. Its length must equal the atom
axis of `atom_mask` and predicted coordinates. The raw ground truth may be
longer before permutation alignment because it retains symmetry-equivalent
atoms; the existing post-alignment shape check must make it match the
prediction. The full array remains internal to structure preprocessing and is
not passed to RASA when token cropping is active.

This does not reduce validation-set coverage: every ID in both validation
manifest partitions is still enumerated. It only keeps each case's metadata
aligned with its configured 768-token crop.

## Failure Handling

An explicitly configured training weight file is a required formal-training
input. Missing IDs, non-finite values, non-positive values, or an unreadable
file raise a clear error instead of silently falling back to uniform
sampling. Validation partitions return uniform placeholder probabilities
without opening the training weight file.

## Verification

Regression tests cover:

- one flat multinomial distribution over multiple datasets;
- raw PT weight preservation and validation isolation;
- identity-preserving cropped AtomArray propagation through collation, matching
  `atom_mask`, acceptance of expanded pre-alignment GT, and rejection of a true
  cropped AtomArray/`atom_mask` mismatch;
- total-loss scaling in both training paths through a shared helper;
- exact runtime YAML sampling and loss settings.

After unit and configuration tests pass, restart the four-GPU run and verify
that sanity validation completes and at least one real training step is
logged.
