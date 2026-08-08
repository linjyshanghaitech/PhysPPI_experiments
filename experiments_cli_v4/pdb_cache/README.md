# pdb_cache (v4) — 预计算 PDB tensors

把 v4 final chunk 的 `gt/` 和 `input/`（全原子 complex PDB）离线转成：

- `{case}_gt.pt`：canonize 后的 GT AtomArray（全重原子，pre-crop）
- `{case}_{chain}_monomer.pt`：从 `input/{case}.pdb` 按 chain 拆分、处理后
  的 monomer AtomArray（全重原子，no-crop + token positions）
- `{case}_seq.pt`：seq_cache 字段 + 推导的 `resseq_to_sequence_index`

用法：

```bash
python generate_pdb_tensors_v4.py \
  --dataset-dir /101045/jinyu/PPI/Prior_out/training/cli_v4/AFDB-10k-1 \
  --output-root /101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4 \
  --dataset-name AFDB-10k-1 \
  --num-workers 8
```

Q-BioLiP 需要 CCD：

```bash
python generate_pdb_tensors_v4.py \
  --dataset-dir /101045/jinyu/PPI/Prior_out/training/cli_v4/Q-BioLiP-10k-new \
  --dataset-name Q-BioLiP-10k-new \
  --ccd-file /101045/jinyu/PhysPPI/assets/openfold3-physppi-20260724/ccd/components.cif
```

注意：
- 不依赖 `.chunk/` 中间产物，monomer 直接从最终 `input/` 按链拆分。
- H 原子在 canonize 时移除，保留全部重原子。
- `save_atomarray_as_pt` / `load_atomarray_from_pt` 复用 v3 的
  `experiments/pdb_cache/pdb_adapter.py`，格式完全兼容。
