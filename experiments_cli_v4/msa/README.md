# msa (v4) — 预计算 MSA tensors

把 `msa_local/{main,paired}/{case}/{case}.npz` 投影到 GT 残基，计算 profile /
deletion_mean，输出 `{case}_msa.pt`，供训练 adapter 直接加载。

输出格式（与 v3 相同）：

```python
{
    "chain_ids": ["A", ...],
    "main_msa": {chain: np.ndarray[N_rows, N_res]},
    "main_deletion": {chain: np.ndarray[N_rows, N_res]},
    "paired_msa": {...},
    "paired_deletion": {...},
    "profile": {chain: np.ndarray[N_res, 32]},
    "deletion_mean": {chain: np.ndarray[N_res]},
    "resseq": {chain: [resseq, ...]},
}
```

用法：

```bash
python generate_msa_tensors_v4.py \
  --dataset-dir /101045/jinyu/PPI/Prior_out/training/cli_v4/AFDB-10k-1 \
  --output-root /101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4 \
  --dataset-name AFDB-10k-1 \
  --chunks chunk0001 \
  --num-workers 8
```

参数：`--chunks`、`--max-samples`、`--num-workers`、`--force`（默认跳过已有）。

注意：
- AFDB 的 MSA 是 uint8 索引，Q-BioLiP 是 `<U1` char，脚本自动转换。
- MSA 行数先截断（main 16384 / paired 8192）再算 profile，避免 Q-BioLiP
  超大 MSA 占满内存。
- npz 内部 key 可能是 `unpaired/{case}` / `paired/{case}` 前缀，由
  `parse_msas_preparsed` 统一处理。
