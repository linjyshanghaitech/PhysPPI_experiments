# PhysPPI OpenFold3 Baseline Inference

## Scope

Baseline inference runs pretrained OpenFold3 with `physppi.enabled=false` on the old PhysPPI test split. This establishes a reference for evaluating PhysPPI-conditioned training.

## Test Set

- Source: old PhysPPI test split (v3-Qbiolip-M)
- Samples: 548 (from v3_split_manifest_clean.json)
- Query JSON: output/physppi_openfold3_baseline/query_test.json
- Small query: output/physppi_openfold3_baseline/query_test_small.json (5 samples)

## Running Baseline Inference

```bash
# Dry run (prints command only):
bash scripts/dev/physppi_run_baseline_inference.sh

# Small test (5 samples):
bash scripts/dev/physppi_run_baseline_inference.sh \
  output/physppi_openfold3_baseline/query_test_small.json --execute

# Full test set (548 samples):
bash scripts/dev/physppi_run_baseline_inference.sh \
  output/physppi_openfold3_baseline/query_test.json --execute
```

## Summarizing Results

```bash
python3 scripts/dev/physppi_summarize_baseline_inference.py \
  --manifest output/physppi_openfold3_baseline/test_set_manifest.json \
  --run-dir output/physppi_openfold3_baseline/runs/small_YYYYMMDD_HHMMSS \
  --output output/physppi_openfold3_baseline/baseline_summary.json
```

## Verification Log

Append results here after each run.


## Small Baseline Run

```text
status: blocked
query_json: output/physppi_openfold3_baseline/query_test_small.json
run_dir: output/physppi_openfold3_baseline/runs/small_20260602_211954
physppi_enabled: false
checkpoint: openfold3-p2-155k
blocker_class: checkpoint_missing_or_download_blocked
evidence_log: output/physppi_openfold3_baseline/runs/small_20260602_211954/inference.log
next_decision: fix the blocker before running full baseline or any training.
```


## Small Baseline Run

```text
status: partial
query_json: output/physppi_openfold3_baseline/query_test_small.json
run_dir: output/physppi_openfold3_baseline/runs/small_20260602_214050
summary_json: output/physppi_openfold3_baseline/baseline_small_summary.json
samples_in_small_query: 5
completed_predictions: 2
failed_predictions: 3 (OOM — GPU out of memory for large complexes)
physppi_enabled: false
checkpoint: openfold3-p2-155k
runner_yaml: examples/example_runner_yamls/physppi_baseline_predict.yml
elapsed_seconds: 3895
gpu_log: output/physppi_openfold3_baseline/runs/small_20260602_214050/gpu.csv
inference_log: output/physppi_openfold3_baseline/runs/small_20260602_214050/inference.log
next_decision: baseline path proven for moderate-sized complexes. Address OOM before scaling up.
```
