# residue_prior (v4) — 预计算 residue prior tensors

从 v4 `prior_pkl/{case}.pkl` 抽取：

- `{case}_interface_residues.pt`：dmap（CA-CA ≤ 18Å）
- `{case}_interaction_pairs.pt`：5 种 PLIP 相互作用（hbond / saltbridge /
  hydrophobic / pistacking / pication）
- `{case}_membrane_relative_location.pt`：膜区间 + 平面（优先 `tmdet`，
  `opm` 兜底）

用法：

```bash
python generate_residue_tensors_v4.py \
  --dataset-dir /101045/jinyu/PPI/Prior_out/training/cli_v4/AFDB-10k-1 \
  --output-root /101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4 \
  --dataset-name AFDB-10k-1 \
  --num-workers 8
```

说明：
- dmap / interactions / membrane 三个 `.pt` 独立存在，缺哪个就不写哪个。
- 所有坐标用 PDB resseq，训练 adapter 直接 `token_index.get((chain, resseq))`。
- `torch.load(..., weights_only=False)`（.pt 里有 tuple 字符串）。
