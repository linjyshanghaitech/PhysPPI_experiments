# Experiments

This directory contains reproducible diagnostics, hypothesis checks, and
not-yet-stable research workflows.

## Dependency direction

- `experiments` may import public APIs from `openfold3`.
- `scripts/physppi` may import entry points from `experiments` and public APIs
  from `openfold3`.
- `openfold3` must never import `experiments` or `scripts`.
- Experiment entry points own only experiment orchestration and provisional
  logic; stable reusable behavior belongs below them.

Use `experiments/` for one-off diagnostics, hypothesis validation, and code
whose interface or evidence is still changing. Put stable operator-facing
commands in `scripts/physppi/`. Put reusable, supported business logic in the
appropriate `openfold3/` package.

## Promotion gate

Promote experiment code only when all six conditions hold:

1. The target problem is reproducible.
2. The fix passes on the target data and regression samples.
3. No hard-coded machine path remains.
4. Reusable logic has formal unit tests.
5. Ruff and the relevant pytest suites pass.
6. The experiment README records the conclusion, configuration, and
   corresponding core implementation commit.

Promotion is move-not-copy. Move reusable implementation into `openfold3/` and
its formal tests into `openfold3/tests/`. Keep only a thin reproduction entry
point, its TOML, and its conclusions under `experiments/`; the thin entry point
must call the promoted API rather than preserve a second implementation.
