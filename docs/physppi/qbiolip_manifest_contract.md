# Q-BioLiP-10k-new manifest contract

`scripts/dev/physppi_snapshot_qbiolip_manifest.py` is the reproducibility
boundary for PhysPPI data. It scans only
`<root>/chunk/chunk*/prior_pkl/*.pkl`; staging trees such as `.chunk` and
legacy cache directories are not traversed.

Run it with:

```bash
python scripts/dev/physppi_snapshot_qbiolip_manifest.py \
  --root /path/to/Q-BioLiP-10k-new \
  --output output/physppi_qbiolip_refactor/qbiolip_manifest.json
```

The JSON payload contains `schema_version`, `dataset_root`, `count` (also
available as `case_count`), and deterministic `rows`, sorted by sample ID.
Each row records the canonical prior, GT complex PDB, monomer PDBs, main MSA
and paired MSA paths, plus SHA-256 values under `files` and
`source_sha256` for the prior. `mandatory_valid` is true only when the prior,
GT PDB, and at least one monomer PDB exist. Missing mandatory paths remain in
the census with a `path_errors` list so an incomplete upstream snapshot cannot
silently look complete.

Main MSA accepts either `<msa-root>/main/<sample>/` or the flat
`<msa-root>/main/` layout. Paired MSA is case-nested at
`<msa-root>/paired/<sample>/`. The snapshot also recognizes these roots under
`msa_local` and beside each chunk. Only exact basenames of the form
`{sample}_{sample}_{chain}.npz` or
`{sample}_{sample}_chain={chain}.npz` are indexed. Chains with no main MSA are
listed in `main_msa_query_only_chains`; this is the mandatory query-only
fallback, not a missing-case error.

Optional prior slots are independent and start unselected. Their state has
`available`, `selected` (always false in a snapshot), `valid` (equal to
`available`), and `provenance` (`qbiolip10k` when available,
`unavailable` otherwise):

| slot | availability rule |
| --- | --- |
| `paired_msa` | at least one canonical paired MSA path |
| `contact` | `bool(prior.dmap)` |
| `interaction` | `bool(prior.interactions)`; empty interactions are unavailable, not negative |
| `surface` | `bool(prior.mesh)` |
| `tm` | `bool(prior.opm.get("res_loc"))` |

Unreadable or non-canonical pickle payloads are retained as rows with all
prior-derived optional slots unavailable and a `prior_load_error` diagnostic.
This keeps the census deterministic while preventing an invalid optional file
from becoming a selected zero tensor downstream.
