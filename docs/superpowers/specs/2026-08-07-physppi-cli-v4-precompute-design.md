# PhysPPI cli_v4 预计算脚本设计

日期：2026-08-07（2026-08-08 根据师姐脚本与用户确认修订）

## 背景与目标

cli_v3 的预计算脚本把 Prior 流程产物（`prior_pkl`、PDB、MSA、`seq_cache`）烘焙成训练时 `torch.load` 直接可用的 `.pt`。本次为 cli_v4 编写对应的预计算脚本，交付物放在 `experiments_cli_v4/` 下同名文件夹中，输出写入 `/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4/`。

**mesh_prior 已由师姐实现**：`experiments_cli_v4/mesh_prior/generate_mesh_tensors_v4.py` + `README.md`。本设计以该脚本为基准：

- 不重写、不修改师姐的 mesh 脚本，只补 `HANDOFF20260808.md`。
- 其余模块（msa、pdb_cache、residue_prior、split_training_val）按同一 CLI / 输出 / 并行风格编写。
- **不写 adapter**。adapter 属于训练侧集成（`_runtime.py`），留到下一步。

**本轮只写脚本和文档，不运行任何生成**（不写 `.pt`、不跑 MMseqs2、不执行生成脚本）。只做 `py_compile` 语法检查和只读数据抽查。

## v4 与 v3 的关键差异（脚本必须适配）

| 维度 | v3 | v4 |
|---|---|---|
| case id | `{case_id}` | `{case_id}_G1`、`_G2` 等分组后缀 |
| 最终结构输入 | `bb_input/{case}_{chain}.pdb`（backbone-only、每链一个文件） | `input/{case}.pdb`（全原子 complex PDB，含 H，整复合物一个文件） |
| mesh 范围 | 仅 interface residues | 整链全残基 |
| 顶点→残基映射 | 脚本自己用 bb 原子做 KDTree | `prior_pkl` 自带 `{case}_point_to_residue`（全重原子 KDTree 结果），缺失时才用 input 全重原子 KDTree 兜底 |
| 膜信息 | `prior_pkl.{case}.opm` | `prior_pkl.{case}.tmdet`（字段兼容 opm） |
| `seq_cache` | 含 `resseq_to_sequence_index` | 不含该字段，需脚本从 input PDB 推导 |
| PDB 原子范围 | backbone-only | 全重原子（去 H 后保留侧链） |

## 统一 CLI / 输出约定（沿用师姐脚本）

每个 generate 脚本提供相同参数：

- `--dataset-dir`：训练根目录，包含 `chunk/`
- `--output-root`：默认 `/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4`
- `--dataset-name`：默认 `AFDB-10k-1`；Q-BioLiP 时传 `Q-BioLiP-10k-new`
- `--chunks`：逗号分隔 chunk 名，默认全部
- `--max-samples`：每 chunk 样本数上限
- `--num-workers`：并行数
- `--force`：覆盖已有 `.pt`，默认跳过已存在

输出布局：

```
{output_root}/{dataset_name}/{chunk_name}/{case_id}_{suffix}.pt
{output_root}/{dataset_name}/{type}_generation_summary.json
```

并行：`multiprocessing.Pool` + `imap_unordered(chunksize=4)`，每个 sample 独立 try/except，失败记入 summary 不中断。

## 输入输出文件

输入根目录：`/101045/jinyu/PPI/Prior_out/training/cli_v4/{dataset}/chunk/{chunk}/`

- `gt/{case}.pdb`：GT complex PDB
- `input/{case}.pdb`：全原子 complex PDB（可能含 H）
- `prior_pkl/{case}.pkl`：`{case: SimpleNamespace(mesh, tmdet, dmap, interactions)}` + `{case}_point_to_residue`
- `msa_local/main/{case}/{case}.npz`、`msa_local/paired/{case}/{case}.npz`（Q-BioLiP 的 msa_local 用户后续补齐）
- `seq_cache.pkl`：per-case per-chain `full_seq_1aa`、`full_seq_3aa`、`mask`、`msa_eligible`

输出根目录：`/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v4/{dataset}/{chunk}/`

`.pt` 命名：

- `{case}_mesh.pt`（师姐脚本已实现）
- `{case}_msa.pt`
- `{case}_gt.pt`
- `{case}_{chain}_monomer.pt`
- `{case}_seq.pt`
- `{case}_interface_residues.pt`
- `{case}_interaction_pairs.pt`
- `{case}_membrane_relative_location.pt`

## 交付物结构

```
experiments_cli_v4/
├── mesh_prior/
│   ├── generate_mesh_tensors_v4.py   # 师姐已写，不改
│   ├── README.md                     # 师姐已写，不改
│   └── HANDOFF20260808.md            # 本次新增
├── msa/
│   ├── generate_msa_tensors_v4.py
│   ├── README.md
│   └── HANDOFF20260808.md
├── pdb_cache/
│   ├── generate_pdb_tensors_v4.py
│   ├── README.md
│   └── HANDOFF20260808.md
├── residue_prior/
│   ├── generate_residue_tensors_v4.py
│   ├── README.md
│   └── HANDOFF20260808.md
└── split_training_val/
    ├── extract_sequences_v4.py
    ├── run_mmseqs_v4.py
    ├── build_split_v4.py
    ├── run_all_v4.py
    ├── README.md
    └── HANDOFF20260808.md
```

## 各模块设计

### mesh_prior

已由师姐脚本实现，本任务不重写。要点（写进 HANDOFF）：

- 读 `point_to_residue`（全重原子映射），缺失时用 `input/{case}.pdb` 全重原子 KDTree 兜底。
- 顶点按最近重原子距离升序，每残基截断 K=64。
- drop hbond，输出 `[N, K, 9]`；行序 = input PDB 每链残基首次出现顺序（与 dmap / point_to_residue 的 resseq 一致）。
- 输出 `{"data": {...}, "resseq": {...}}`。

### msa

输入：`msa_local/{main,paired}/{case}/{case}.npz`、`gt/{case}.pdb`、`seq_cache.pkl`。

1. 移植 v3 `generate_msa_tensors.py` 的 MsaArray 读取、char 转换、deletion、profile、deletion_mean 计算。
2. 适配 v4 npz 布局：文件位于 `msa_local/{main,paired}/{case}/`，npz 内部 key 为 `unpaired/{原 case id}` / `paired/{原 case id}`（不带 `_G1`）；解析时兼容 `unpaired/`、`paired/` 前缀以及裸 key。
3. 按 GT PDB 每链序列 + resseq 做投影（沿用 v3 `_project_msa_to_gt_residues`），分组 case 各自投影。
4. `profile` / `deletion_mean` 从未截断 main MSA 计算。
5. 输出 `{case}_msa.pt`：`chain_ids`、`main_msa`、`main_deletion`、`paired_msa`、`paired_deletion`、`profile`、`deletion_mean`、`resseq`。
6. Q-BioLiP `msa_local` 用户后续补齐，脚本按完整布局编写。

### pdb_cache

输入：`gt/{case}.pdb`、`input/{case}.pdb`、`seq_cache.pkl`；Q-BioLiP 需要 CCD。

1. `{case}_gt.pt`：`gt/{case}.pdb` canonize 后保存**全重原子**，保留所有 annotation；格式与 v3 `save_atomarray_as_pt` / `load_atomarray_from_pt` 一致。
2. `{case}_{chain}_monomer.pt`：从 `input/{case}.pdb` 按 chain 切分（去 H、保留全重原子），每链独立 AtomArray，再做 entity id 标注、`process_target_atom_array_of3`（no-crop）、token positions、crop_mask 等处理。不依赖 `.chunk/` 中间目录。
3. `{case}_seq.pt`：保存 v4 seq_cache 字段（`full_seq_1aa`、`full_seq_3aa`、`mask`、`msa_eligible`），并写入脚本推导的 `resseq_to_sequence_index`（由 input PDB 链残基顺序与 seq_cache 对齐得到）。
4. 不 import `_runtime.py`，canonicalization / monomer processing 逻辑 inline 重写（同 v3 做法）。

### residue_prior

输入：`prior_pkl/{case}.pkl`。

1. `{case}_interface_residues.pt`：`dmap`，格式同 v3。
2. `{case}_interaction_pairs.pt`：`interactions`，只保留 5 种类型，保留 atom names。
3. `{case}_membrane_relative_location.pt`：优先读 `tmdet`（`res_loc`、`outer_centroid`、`outer_normal`、`thicken`），fallback 读 `opm`。

### split_training_val

1. `extract_sequences_v4.py`：从 v4 `seq_cache.pkl` 提取 chain 序列（header `>{case}|{chain}`）。
2. `run_mmseqs_v4.py`：MMseqs2 `easy-cluster`，30% 和 40% identity，沿用 v3 参数。
3. `build_split_v4.py`：chain-level 同源图 → 连通分量 → 贪心分配，9:1，seed 42；生成 `split_manifest.json`、`sampling_weights.pt`、`train_ids.json`、`val_ids.json`、`split_report.json`。
4. `run_all_v4.py`：一键执行三步。
5. 输出复制到 `precomputed_mesh/cli_v4/`（`split_manifest.json` + `sampling_weights.pt`）。
6. 沿用 v3 的 inventory reconciliation 概念：把 manifest 中预计算/reader 不存在的 case 记录为 removed（不重新分 split）。

## 错误处理

- 每个 sample 独立 try/except，失败记入 summary，不中断整个 chunk。
- 输出目录在写入前创建；`torch.save` 先写 `.tmp` 再 `replace`。
- `torch.load(..., weights_only=False)`。
- `save_atomarray_as_pt` 保留 numpy dtype（字符串 annotation 不做 `.tolist()`）。
- 生成脚本不 import `_runtime.py`（避免 CUDA 依赖链）。

## 验证（本轮只做只读）

- 全部 `.py` 通过 `py_compile.compile(..., doraise=True)`。
- 只读抽查 v4 数据假设：prior_pkl key/字段、input PDB 链划分、seq_cache 对齐、MSA npz key 形态。
- 不写 `precomputed_mesh/cli_v4/` 任何文件。

## 明确不在本次范围

- 不改师姐的 `generate_mesh_tensors_v4.py` / `README.md`。
- 不写 adapter，不改 `openfold3/` 下的 `_runtime.py` / reader。
- 不运行任何生成（`.pt`、MMseqs2、全量或单 chunk）。
- 不修改 v3 的 `experiments/` 内容。
- 不动 `precomputed_mesh/cli_v4/AFDB-10k-1-old`（师姐的 mesh 试点产物）。

## 本地数据状态备注

- AFDB-10k-1 v4：50+ chunk 基本完整（10000 input / 10051 prior_pkl），可用于后续生成验证。
- Q-BioLiP-10k-new v4：本地仅 chunk0004 有 267 sample，`msa_local` 待用户补齐；脚本按完整布局编写。
- `precomputed_mesh/cli_v4/AFDB-10k-1`、`Q-BioLiP-10k-new` 当前为空，等待生成。
