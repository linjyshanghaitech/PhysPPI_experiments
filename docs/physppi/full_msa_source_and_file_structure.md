# PhysPPI OF3 Full MSA Source And File Structure

Date: 2026-06-05

## Current Status

This note records the current full-source1 protein-protein cache state for
`/101045/jinyu/PhysPPI/openfold-3`.

Per the current execution standard, the engineering target is treated as
complete because the only remaining blocker is the asynchronous full-dataset
MSA search. The post-MSA cache/dataset gate should still wait for the final MSA
report before building alignment arrays, the training cache, the dataset config,
and the full feature sanity report.

Observed status:

- Repository branch: `main`
- Latest recorded commit: `fa5a1c76 Record one-MSA PhysPPI overfit smoke`
- Full source1 active manifest:
  `output/physppi_full_msa_cache/full_source1_of3_compatible_manifest.json`
- Active sample count: `3682`
- Protein scope: protein-protein only
- Interface source: `interface_pair_source=1`
- MSA process PID: `498447`
- MSA chunks: `231` chunks, `max_queries_per_chunk=16`
- Passed chunks at last check: `9` (`chunk_0000` through `chunk_0008`)
- Final full MSA report: not yet present
  (`output/physppi_full_msa_cache/colabfold_msas/chunked_msa_report.json`)

## Which MSA Databases Are Being Searched

The current run uses the OpenFold3 `align-msa-server` path, not local
JackHMMER/HHblits database binaries.

Actual runtime entry:

```bash
python -m openfold3.run_openfold align-msa-server \
  --query-json <chunk_query_json> \
  --output-dir <chunk_colabfold_msas_dir> \
  --msa-computation-settings-yaml output/physppi_full_msa_cache/msa_server_full.yml
```

The MSA settings file is:

```yaml
msa_file_format: npz
server_user_agent: openfold-physppi-full-source1
server_url: https://api.colabfold.com
save_mappings: true
cleanup_msa_dir: false
```

Therefore, the current search is through the ColabFold/MMseqs2 API at
`https://api.colabfold.com`.

From `openfold3/core/data/tools/colabfold_msa_server.py`, the unpaired/main MSA
path uses `use_pairing=False`, `use_env=True`, and `use_filter=True`, which
requests mode `env`. The downloaded unpaired files are:

- `uniref.a3m`
- `bfd.mgnify30.metaeuk30.smag30.a3m`

The paired MSA path uses `use_pairing=True`, `use_env=True`, and greedy pairing,
which requests mode `pairgreedy-env`. Its downloaded paired file is:

- `pair.a3m`

Template search is not enabled in the current PhysPPI full-source1 MSA run:
`query_format_main()` calls `query_colabfold_msa_server(..., use_templates=False)`.
So this run should not be treated as a PDB70-template search, even though the
ColabFold utility has code paths that can parse `pdb70.m8` when template mode is
enabled.

For comparison only, the repository also contains
`scripts/snakemake_msa/download_of3_databases.py`, which can download local
alignment databases:

- JackHMMER defaults: `uniprot`, `uniref90`, `mgnify`, `pdb_seqres`
- HHblits default: `uniref30`
- Optional protein databases: `bfd`, `cfdb`
- Optional RNA databases when explicitly requested

Those local database names are not the source of the current full-source1 run.
The current run is ColabFold/MMseqs2 server based.

## If Existing MSAs Are Available

Existing MSAs can potentially replace the slow server step if they are converted
into the OpenFold3 ColabFold output contract expected by the downstream scripts.

The downstream contract is:

- Main/unpaired per representative sequence:
  `output/physppi_full_msa_cache/colabfold_msas/main/<representative_id>.npz`
- Paired per complex and representative sequence:
  `output/physppi_full_msa_cache/colabfold_msas/paired/<complex_id>/<representative_id>.npz`
- Mappings:
  `output/physppi_full_msa_cache/colabfold_msas/mappings/*.json`
- Final gate report:
  `output/physppi_full_msa_cache/colabfold_msas/chunked_msa_report.json`

Each `.npz` is produced by parsing A3M with OF3's `parse_a3m()` and saving the
resulting dict. The expected keys used in this pipeline are:

- Main: `colabfold_main`
- Paired: `colabfold_paired`

If you have existing A3M files rather than OF3-ready `.npz` files, the cleanest
adapter is to reuse the same parser path in
`openfold3/core/data/tools/colabfold_msa_server.py`:

```python
msas = {"colabfold_main": parse_a3m(a3m_text)}
np.savez_compressed(npz_file, **{k: v.to_dict() for k, v in msas.items()})
```

and analogously for `colabfold_paired`.

Do not skip the mapping files. The chunk merge path requires:

- `rep_id_to_seq.json`
- `seq_to_rep_id.json`
- `chain_id_to_rep_id.json`
- `query_name_to_complex_id.json`

These are used to connect query chains, representative sequence IDs, complex
IDs, and final per-chain MSA paths.

## File Structure

Current root:

```text
output/physppi_full_msa_cache/
├── full_source1_of3_compatible_manifest.json
├── full_source1_manifest.json
├── full_cache_input_manifest.json
├── full_cache_input_plan.json
├── query_full_source1.json
├── sample_to_structure_id.json
├── msa_server_full.yml
├── start_chunked_msa.sh
├── chunked_msa.pid
├── chunked_msa.logpath
├── logs/
│   └── chunked_msa_*.log
├── stage_status/
│   ├── stage_cifs.json
│   ├── stage_protein_only_cifs.json
│   ├── preprocess_unique_structures.json
│   ├── filter_manifest_by_preprocessed_fastas.json
│   ├── build_query_from_preprocessed_fastas.json
│   └── align_msa_server.json
├── protein_only_cif/
│   └── <sample_id>.cif
├── preprocessed_pdb/
│   ├── chunked_preprocess_report.json
│   ├── chunk_status/
│   │   └── chunk_*.json
│   ├── reference_mols/
│   │   └── <CCD>.sdf
│   └── structure_files/
│       └── <sample_id>/
│           └── OF3 preprocessed structure artifacts
└── colabfold_msas/
    ├── chunks/
    │   ├── query_json/
    │   │   └── chunk_0000.json ... chunk_0230.json
    │   └── chunk_0000/
    │       └── colabfold_msas/
    │           ├── main/
    │           ├── paired/
    │           ├── raw/
    │           └── mappings/
    └── chunk_status/
        └── chunk_*.json
```

After all chunks pass, `scripts/dev/physppi_run_chunked_msa_server.py` merges
chunk outputs into:

```text
output/physppi_full_msa_cache/colabfold_msas/
├── main/
│   └── <representative_id>.npz
├── paired/
│   └── <complex_id>/
│       └── <representative_id>.npz
├── mappings/
│   ├── README.md
│   ├── rep_id_to_seq.json
│   ├── seq_to_rep_id.json
│   ├── chain_id_to_rep_id.json
│   └── query_name_to_complex_id.json
└── chunked_msa_report.json
```

Those merged outputs are the prerequisite for the remaining post-MSA stages:

```text
output/physppi_full_msa_cache/
├── alignment_arrays/
├── training_cache_full_source1.json
├── dataset_config_full_source1.json
└── feature_sanity_full_source1.json
```

## Next Gate After MSA Completion

Once `chunked_msa_report.json` exists and reports `status=passed`,
`failed_chunk_count=0`, and `max_queries_per_chunk=16`, run the post-MSA gate in
this order:

1. `build_alignment_arrays`
2. `build_training_cache`
3. `build_dataset_config`
4. Full-source1 feature sanity report

Do not start full-dataset fine-tuning before the dataset config and feature
sanity report pass on the merged full-source1 outputs.
