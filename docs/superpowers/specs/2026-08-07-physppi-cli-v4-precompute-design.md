# PhysPPI cli_v4 预计算脚本设计

日期：2026-08-07

## 背景与目标

cli_v3 的预计算脚本（`experiments/mesh_prior`、`msa`、`pdb_cache`、`residue_prior`、`split_training_val`）把 Prior 流程产物（`prior_pkl`、PDB、MSA、`seq_cache`）烘焙成训练时 `torch.load` 直接可用的 `.pt`。本次为 cli_v4 编写对应的预计算脚本，交付物放在 `experiments_cli_v4/` 下同名 5 个文件夹中，输出写入 `/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4/`。

**本轮只写脚本和文档，不运行任何生成**（不写 `.pt`、不跑 MMseqs2、不执行生成脚本）。只做 `py_compile` 语法检查和只读数据抽查。

## v4 与 v3 的关键差异（脚本必须适配）

| 维度 | v3 | v4 |
|---|---|---|
| case id | `{case_id}` | `{case_id}_G1`、`_G2` 等分组后缀 |
| 最终结构输入 | `bb_input/{case}_{chain}.pdb`（backbone-only、每链一个文件） | `input/{case}.pdb`（全原子 complex PDB，含 H，整复合物一个文件） |
| mesh 范围 | 仅 interface residues | 整链全残基 |
| 顶点→残基映射 | 脚本自己用 bb 原子做 KDTree | `prior_pkl` 自带 `{case}_point_to_residue`（全重原子 KDTree 结果） |
| 膜信息 | `prior_pkl.{case}.opm` | `prior_pkl.{case}.tmdet`（字段兼容 opm） |
| `seq_cache` | 含 `resseq_to_sequence_index` | 不含该字段，需脚本从 input PDB 推导 |
| PDB 原子范围 | backbone-only | 全重原子（去 H 后保留侧链） |

## 统一 I/O 约定

输入根目录：`/101045/jinyu/PPI/Prior_out/training/cli_v4/{dataset}/chunk/{chunk}/`

- `gt/{case}.pdb`：GT complex PDB
- `input/{case}.pdb`：全原子 complex PDB（可能含 H）
- `prior_pkl/{case}.pkl`：`{case: SimpleNamespace(mesh, tmdet, dmap, interactions)}` + `{case}_point_to_residue`
- `msa_local/main/{case}/{case}.npz`、`msa_local/paired/{case}/{case}.npz`
- `seq_cache.pkl`：per-case per-chain `full_seq_1aa`、`full_seq_3aa`、`mask`、`msa_eligible`

输出根目录：`/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4/{dataset}/{chunk}/`

`.pt` 命名与 v3 一致：

- `{case}_mesh.pt`
- `{case}_msa.pt`
- `{case}_gt.pt`
- `{case}_{chain}_monomer.pt`
- `{case}_seq.pt`
- `{case}_interface_residues.pt`
- `{case}_interaction_pairs.pt`
- `{case}_membrane_relative_location.pt`

每个数据集根目录写 `generation_summary.json`、`pdb_generation_summary.json` 等统计（沿用 v3 模式）。

## 交付物结构

```
experiments_cli_v4/
├── mesh_prior/
│   ├── generate_mesh_tensors.py
│   ├── mesh_adapter.py
│   ├── README.md
│   └── HANDOFF20260807.md
├── msa/
│   ├── generate_msa_tensors.py
│   ├── msa_adapter.py
│   ├── README.md
│   └── HANDOFF20260807.md
├── pdb_cache/
│   ├── generate_pdb_tensors.py
│   ├── pdb_adapter.py
│   ├── README.md
│   └── HANDOFF20260807.md
├── residue_prior/
│   ├── generate_residue_tensors.py
│   ├── residue_adapter.py
│   ├── README.md
│   └── HANDOFF20260807.md
└── split_training_val/
    ├── extract_sequences.py
    ├── run_mmseqs.py
    ├── build_split.py
    ├── run_all.py
    ├── README.md
    └── HANDOFF20260807.md
```

## 各模块设计

### mesh_prior

输入：`prior_pkl/{case}.pkl`、`input/{case}.pdb`、`gt/{case}.pdb`、`seq_cache.pkl`。

1. 从 pkl 读取 `{case}_point_to_residue`（`{chain: [resseq_per_vertex, ...]}`）和 `{case}.mesh`（`{chain: [[x,y,z,charge,hbond,hphob,nx,ny,nz,si], ...]}`）。
2. 优先使用 `point_to_residue`；若缺失，fallback 从 `input/{case}.pdb` 按链取全重原子（去 H）建 cKDTree，对每个 mesh 顶点做最近重原子映射（与 v4 pipeline 的 `compute_point_to_residue` 等价）。
3. 顶点 10 列 → `.pt` 9 列：drop hbond（保留 `xyz、charge、hphob、normal、si`），与 v3 相同。
4. 对每条链：`data[N, K=64, 9]`，N = `seq_cache` 该链 `full_seq_3aa` 长度；K 为参数，默认 64。
5. v4 `seq_cache` 没有 `resseq_to_sequence_index`：先用 input PDB 每链的 (resseq, resname) 与 seq_cache 做 greedy 三字母顺序匹配构建 `resseq → seq_index` 映射（沿用 v3 fallback 逻辑）；对不上的 seq 行保持全零、`resseq=-1`。
6. 输出 `{case}_mesh.pt`：`{"data": {chain: Tensor[N, K, 9]}, "resseq": {chain: list[int]}}`。

### msa

输入：`msa_local/{main,paired}/{case}/{case}.npz`、`gt/{case}.pdb`、`seq_cache.pkl`。

1. 移植 v3 `generate_msa_tensors.py` 的 MsaArray 读取、char 转换、deletion、profile、deletion_mean 计算。
2. 适配 v4 npz 布局：文件位于 `msa_local/{main,paired}/{case}/`，npz 内部 key 为 `unpaired/{原 case id}` / `paired/{原 case id}`（不带 `_G1`）；解析时兼容 `unpaired/`、`paired/` 前缀以及裸 key。
3. 按 GT PDB 每链序列 + resseq 做投影（沿用 v3 `_project_msa_to_gt_residues`），分组 case 各自投影。
4. `profile` / `deletion_mean` 从未截断 main MSA 计算。
5. 输出 `{case}_msa.pt`：`chain_ids`、`main_msa`、`main_deletion`、`paired_msa`、`paired_deletion`、`profile`、`deletion_mean`、`resseq`。

### pdb_cache

输入：`gt/{case}.pdb`、`input/{case}.pdb`、`seq_cache.pkl`；Q-BioLiP 需要 CCD。

1. `{case}_gt.pt`：`gt/{case}.pdb` canonize 后保存**全重原子**（去 H、MSE→MET、FME 修正、CCD canonicalize），保留所有 annotation；格式与 v3 `save_atomarray_as_pt` / `load_atomarray_from_pt` 一致。
2. `{case}_{chain}_monomer.pt`：从 `input/{case}.pdb` 按 chain 切分（去 H、保留全重原子），每链独立 AtomArray，再做 entity id 标注、`process_target_atom_array_of3`（no-crop）、token positions、crop_mask 等处理，保存为 `{case}_{chain}_monomer.pt`。不依赖 `.chunk/` 中间目录。
3. `{case}_seq.pt`：保存 v4 seq_cache 字段（`full_seq_1aa`、`full_seq_3aa`、`mask`、`msa_eligible`），并写入脚本推导的 `resseq_to_sequence_index`（由 input PDB 链残基顺序与 seq_cache 对齐得到）。
4. 不 import `_runtime.py`，canonicalization / monomer processing 逻辑 inline 重写（同 v3 做法），并在文档中注明需与上游同步。

### residue_prior

输入：`prior_pkl/{case}.pkl`。

1. `{case}_interface_residues.pt`：`dmap`（`A|B` → `[[r_a, r_b, dist], ...]`），格式同 v3。
2. `{case}_interaction_pairs.pt`：`interactions`，只保留 5 种类型（hbond/saltbridge/hydrophobic/pistacking/pication），保留 atom names。
3. `{case}_membrane_relative_location.pt`：优先读 `tmdet`（`res_loc`、`outer_centroid`、`outer_normal`、`thicken`），fallback 读 `opm`。

### split_training_val

1. `extract_sequences.py`：从 v4 `seq_cache.pkl` 提取 chain 序列（header `>{case}|{chain}`）。
2. `run_mmseqs.py`：MMseqs2 `easy-cluster`，30% 和 40% identity，沿用 v3 参数。
3. `build_split.py`：chain-level 同源图 → 连通分量 → 贪心分配，9:1，seed 42；生成 `split_manifest.json`、`sampling_weights.pt`、`train_ids.json`、`val_ids.json`、`split_report.json`。
4. `run_all.py`：一键执行三步。
5. 输出复制到 `precomputed_mesh/cli_v4/`（`split_manifest.json` + `sampling_weights.pt`）。
6. 沿用 v3 的 inventory reconciliation 概念：把 manifest 中预计算/reader 不存在的 case 记录为 removed（不重新分 split）。

## 错误处理

- 每个 sample 独立 try/except，失败记入 summary，不中断整个 chunk。
- 输出目录必须在主进程先创建，再启动 chunk 级 `ProcessPoolExecutor`。
- `torch.load(..., weights_only=False)`。
- `save_atomarray_as_pt` 保留 numpy dtype（字符串 annotation 不做 `.tolist()`）。
- 生成脚本不 import `_runtime.py`（避免 CUDA 依赖链）。

## 验证（本轮只做只读）

- 全部 `.py` 通过 `py_compile.compile(..., doraise=True)`。
- 只读抽查 v4 数据假设：prior_pkl key/字段、input PDB 链划分、seq_cache 对齐、MSA npz key 形态。
- 不写 `precomputed_mesh/cli_v4/` 任何文件。

## 明确不在本次范围

- 不改 `openfold3/` 下的 `_runtime.py` / reader（训练侧 v4 集成是下一步）。
- 不运行任何生成（`.pt`、MMseqs2、全量或单 chunk）。
- 不修改 v3 的 `experiments/` 内容。

## 本地数据状态备注

- AFDB-10k-1 v4：50 chunk 基本完整（9990 input / 10040 prior_pkl），可用于后续生成验证。
- Q-BioLiP-10k-new v4：本地仅 chunk0004 有 267 sample 且无 `msa_local`，脚本按完整布局编写，但全量生成需等待数据到位。
- `precomputed_mesh/cli_v4/` 当前为空目录。
