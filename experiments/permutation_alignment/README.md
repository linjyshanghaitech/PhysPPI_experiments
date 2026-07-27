# Permutation alignment experiments

Run these entry points through `scripts/physppi/submit_experiment.py` from the
repository root.

| Entry point | Configuration | Primary output |
|---|---|---|
| `experiments.permutation_alignment.audit_protected_chain_matching` | `configs/experiments/permutation_alignment/audit_protected_chain_matching.toml` | `audit-protected-chain-matching-v1/audit.jsonl` |
| `experiments.permutation_alignment.build_minimal_alignment_arrays` | `configs/experiments/permutation_alignment/build_minimal_alignment_arrays.toml` | `build-minimal-arrays-v1/alignment_representatives.fasta`, `alignment_arrays/`, and `alignment_array_report.json` |
| `experiments.permutation_alignment.audit_v1_v32_data_errors` | `configs/experiments/permutation_alignment/audit_v1_v32_data_errors.toml` | `audit-v1-v32-data-errors-v1/data-error-ids.json` |

Each output path is relative to
`${OUTPUT_ROOT}/experiments/permutation_alignment/`. The committed TOML files
declare every required external input; machine-specific roots are supplied by
environment variables.

Every conclusion drawn from these experiments must cite the core implementation
commit that was evaluated. A configuration commit alone is insufficient to
identify the behavior under test.
