# Local experiment overrides

Files named `configs/local/*.toml` contain site- or user-specific values and are
ignored by Git and are never discovered automatically.
`configs/local/example.toml` is not provided by Git. Before running the
launcher, create that ignored file from this template:

```toml
schema_version = 1

[slurm]
partition = "site-partition"
account = "site-account"
qos = "site-qos"
nodes = 1
ntasks_per_node = 1
cpus_per_task = 8
gpus_per_node = 1
memory = "64G"
time = "04:00:00"

[environment]
required = ["PHYSPPI_MSA_CACHE", "PHYSPPI_OUTPUT_ROOT"]

[environment.export]
PHYSPPI_SITE_CACHE = "/path/to/local/cache"
```

A local override may contain only `schema_version = 1`, `[slurm]`, and
`[environment]`. After creating the file, load it explicitly with
`--local-config`:

```bash
PHYSPPI_MSA_CACHE=/path/to/msa-cache \
PHYSPPI_OUTPUT_ROOT=/path/to/output \
python scripts/physppi/submit_experiment.py \
  configs/experiments/permutation_alignment/build_minimal_alignment_arrays.toml \
  --local-config configs/local/example.toml
```

Allowed Slurm keys are `partition`, `account`, `qos`, `nodes`,
`ntasks_per_node`, `cpus_per_task`, `gpus_per_node`, `memory`, and `time`.
The environment section may replace `required` and define `export` values.
Local overrides cannot change the experiment module, arguments, run name,
artifact directory, or runner YAML.
