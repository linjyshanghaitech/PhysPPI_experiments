# mesh_prior (v4) — 预计算 per-residue mesh tensors

把 v4 的 `prior_pkl`(full-chain mesh + `point_to_residue` 映射)转换为
每残基固定形状的 `.pt` 文件,供训练 adapter 或后续分析直接加载。

## 转换方法

与 `experiments/mesh_prior/generate_mesh_tensors.py`(v3)一致:

1. 每个 mesh 顶点归属到**距离最近的重原子所在的残基**(KDTree,单点单残基;
   只使用重原子,排除 H/D/T 与 HOH);
2. v4 PKL 已预计算此映射(`{case_id}_point_to_residue`,1-based PDB resseq),
   脚本优先直接读取;缺失时用 `chunk*/input/{case_id}.pdb` 的全原子坐标重算
   (已验证两种方式逐顶点一致);
3. 顶点按“到最近重原子的距离”升序排列(closest-first),每残基截断到 K=64;
4. 去掉 hbond 列,输出 9 列:`x,y,z,charge,hphob,nx,ny,nz,si`;
5. 行顺序 = 输入 PDB 每链残基的首次出现顺序(与 dmap / point_to_residue 的
   resseq 编号一致)。

## 输出格式

```python
{
    "data":   {chain_id: torch.Tensor[N, K, 9]},   # 无顶点的残基行为全零
    "resseq": {chain_id: [resseq_0, resseq_1, ...]},
}
```

## 用法

```bash
python generate_mesh_tensors_v4.py \
  --dataset-dir /101045/jinyu/PPI/Prior_out/training/cli_v4/AFDB-10k-1-old \
  --output-root /101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4 \
  --dataset-name AFDB-10k-1 \
  --chunks chunk0009 \
  --num-workers 8
```

常用参数:`--chunks`(逗号分隔,默认全部)、`--max-samples`、`--k`、
`--force`(覆盖已有文件,默认跳过已存在)、`--num-workers`。

输出布局:

```
/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4/AFDB-10k-1/
  chunk0001/{case_id}_mesh.pt
  ...
  chunk0050/{case_id}_mesh.pt
  generation_summary.json
```

## 校验

试点(chunk0009,200 samples)已核对:pt 中每残基 charge/hphob/si 均值与直接
由 `prior_pkl` 的 mesh + `point_to_residue` 计算的结果一致(最大差 1e-6,
浮点精度;仅 >64 顶点的残基按最近 64 个顶点截断)。
