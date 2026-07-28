# PDB Cache — 预计算 PDB / Seq 张量

## 目的

PhysPPI 训练每个 sample 都要做 PDB 文本解析 + 原子标准化（去 H、MSE→MET、FME 修正、CCD 处理）。
这些操作是纯确定性的，每个 epoch 重复做纯属浪费 I/O 和 CPU。

**pdb_cache** 离线完成解析 + 标准化，存为 `.pt` 文件。训练时直接 `torch.load`，
**跳过全部 PDB 文本 I/O 和原子归一化**。

## 产出文件

每个 sample 生成 3 类文件，与已有 mesh / residue prior `.pt` 放在同一目录树下：

```
precomputed_mesh/cli_v3/{dataset}/{chunk}/
├── {case_id}_gt.pt              ← GT PDB, canonized AtomArray (complex coords)
├── {case_id}_A_monomer.pt       ← monomer PDB chain A, processed AtomArray
├── {case_id}_B_monomer.pt       ← monomer PDB chain B
└── {case_id}_seq.pt             ← seq_cache 序列元数据
```

---

## 文件格式展开

### 1. `{case_id}_gt.pt`

GT PDB 经过解析 + 链限定 + 重原子标准化 + entity ID 标注后的
**完整 `biotite.structure.AtomArray`**，以 per-atom annotation 数组的形式存储。

**停在 `process_target_atom_array_of3` 之前**，因为 crop/tokenize 依赖运行时配置。

**实际内容（以 `AF-0000000065760170-model_v1` 为例，1320 KB）：**

```python
torch.load("AF-..._gt.pt", weights_only=False)

# 返回一个 dict，key 是 annotation 名，value 是 numpy ndarray：
{
    # ── 坐标（唯一存为 torch.Tensor 的字段）──
    "coord":          torch.Tensor([4872, 3]),   # float32, complex 坐标系

    # ── 核心标识（per-atom 数组，长度 = N_atoms = 4872）──
    "chain_id":       ndarray(["A","A","A",...]),     # dtype U4, 链标识
    "res_id":         ndarray([1,1,1,1,1,1,1,1, 2,2,2,2, 3,...]),  # int64, PDB 原始残基号
    "res_name":       ndarray(["MET","MET","MET",...]), # dtype U5, 三字母残基名
    "atom_name":      ndarray(["N","CA","C","CB","O","CG","SD","CE", "N","CA","C","O", ...]),  # dtype U6
    "element":        ndarray(["N","C","C","C","O","C","S","C", ...]),  # dtype U2

    # ── 训练必需 ──
    "entity_id":      ndarray([1,1,1,...]),            # int32
    "mol_entity_id":  ndarray([1,1,1,...]),            # int32
    "molecule_type_id": ndarray([0,0,0,...]),          # int64, 0=PROTEIN
    "occupancy":      ndarray([1.0,1.0,1.0,...]),      # float64

    # ── PDB 元信息（parse_pdb_af2 自动填充）──
    "atom_id":        ndarray([1,2,3,...]),            # int64
    "auth_asym_id":   ndarray(["A","A","A",...]),      # dtype U4
    "auth_atom_id":   ndarray(["N","CA","C",...]),     # dtype U6
    "auth_comp_id":   ndarray(["MET","MET","MET",...]),# dtype U5
    "auth_seq_id":    ndarray([1,1,1,...]),            # int64
    "label_asym_id":  ndarray(["A","A","A",...]),      # dtype U4
    "label_atom_id":  ndarray(["N","CA","C",...]),     # dtype U6
    "label_comp_id":  ndarray(["MET","MET","MET",...]),# dtype U5
    "charge":         ndarray([0,0,0,...]),            # int64
    "hetero":         ndarray([False,False,...]),      # bool
    "ins_code":       ndarray(["","","",...]),         # dtype U1
}

# 残基复原方式（不需要额外存储 index）：
import biotite.structure as struc
starts = struc.get_residue_starts(atom_array)   # 扫描 res_id + chain_id 找边界
# starts = [0, 8, 12, 20, 28, 36, ...]
#            ↑ MET 8 atoms
#                ↑ GLY 4 atoms
#                    ↑ ASP 8 atoms
```

**per-residue 展开（Chain A 前 3 个残基）：**

```
[0] chain=A  res_id=1  res_name=MET
    atom=N     coord=(  39.469   11.508   13.656)
    atom=CA    coord=(  39.469   10.359   14.562)
    atom=C     coord=(  38.344   10.461   15.586)
    atom=CB    coord=(  40.812   10.258   15.273)
    atom=O     coord=(  38.438   11.211   16.562)
    atom=CG    coord=(  41.844    9.430   14.523)
    atom=SD    coord=(  43.438    9.297   15.430)
    atom=CE    coord=(  44.062    7.738   14.742)

[1] chain=A  res_id=2  res_name=GLY
    atom=N     coord=(  37.062   10.750   15.273)
    atom=CA    coord=(  36.094   11.203   16.266)
    atom=C     coord=(  36.031   10.289   17.484)
    atom=O     coord=(  36.312    9.094   17.391)

[2] chain=A  res_id=3  res_name=ASP
    atom=N     coord=(  36.531   10.766   18.625)
    atom=CA    coord=(  36.531   10.039   19.875)
    atom=C     coord=(  35.375    9.070   19.969)
    atom=CB    coord=(  36.531   11.008   21.062)
    atom=O     coord=(  34.219    9.492   19.906)
    atom=CG    coord=(  37.844   11.094   21.781)
    atom=OD1   coord=(  38.750   10.273   21.500)
    atom=OD2   coord=(  38.000   11.984   22.641)
```

**Chain 统计：**

```
Chain A: 2436 atoms, 312 residues, res_id range 1-312
Chain B: 2436 atoms, 312 residues, res_id range 1-312
```

GT 保存完整 sidechain 原子（每残基 4~14 atoms），各氨基酸原子类型：

```
MET: N CA C O CB CG SD CE             (8 atoms)
GLY: N CA C O                          (4 atoms)
ASP: N CA C O CB CG OD1 OD2            (8 atoms)
TRP: N CA C O CB CG CD1 CD2 CE2 CE3 CZ2 CZ3 CH2 NE1  (14 atoms)
...
共 36 种原子类型，覆盖全部 20 种标准氨基酸
```

**即使 PDB 残基号不从 1 开始，`res_id` 也是原始值：**

```
# 1a0s_1 (Q-BioLiP): 三个 chain 的 residue 都是从 71 开始
Chain A: 413 residues, res_id range 71-483
  SER-71  GLY-72  PHE-73  GLU-74  ...  THR-481  TRP-482  PHE-483
```

---

### 2. `{case_id}_{chain}_monomer.pt`

Monomer PDB 经过完整处理（含 `process_target_atom_array_of3` no-crop + `add_token_positions`）的 AtomArray。

**实际内容（以 `AF-..._A_monomer.pt` 为例，427 KB）：**

```python
torch.load("AF-..._A_monomer.pt", weights_only=False)

{
    # ── 坐标 ──
    "coord":            torch.Tensor([1248, 3]),  # float32, monomer-local 坐标系

    # ── 核心标识（per-atom）──
    "chain_id":         ndarray(["A","A",...]),        # dtype U4
    "res_id":           ndarray([1,1,1,1, 2,2,2,2, ...]),  # int64, PDB 原始残基号
    "res_name":         ndarray(["MET","MET","MET","MET", "GLY","GLY","GLY","GLY", ...]),
    "atom_name":        ndarray(["N","CA","C","O", "N","CA","C","O", ...]),  # 只有 backbone!
    "element":          ndarray(["N","C","C","O", ...]),

    # ── 训练必需 ──
    "entity_id":        ndarray([1,1,...]),            # int32
    "mol_entity_id":    ndarray([1,1,...]),            # int64
    "molecule_type_id": ndarray([0,0,...]),            # int64, 0=PROTEIN
    "occupancy":        ndarray([1.0,1.0,...]),        # float64

    # ── 处理后的 annotation（process_target_atom_array_of3 + add_token_positions 添加）──
    "token_id":         ndarray([0,0,0,0, 1,1,1,1, 2,2,2,2, ...]),  # int64, 残基级 token ID
    "token_position":   ndarray([0,0,0,0, 1,1,1,1, 2,2,2,2, ...]),  # int64
    "is_atomized":      ndarray([False,False,...]),      # bool, PhysPPI 全为 False
    "crop_mask":        ndarray([True,True,...]),         # bool, no-crop 全为 True
    "component_id":     ndarray([1,1,...]),              # int64
    "token_center_atom": ndarray([False,True,False,False, ...]),  # CA=True

    # ── 对称性相关（assign_mol_permutation_ids 添加）──
    "mol_sym_id":           ndarray([1,1,...]),          # int64
    "mol_sym_component_id": ndarray([1,1,...]),          # int64
    "mol_sym_token_index":  ndarray([0,0,...]),          # int64

    # ── PDB 元信息 + 原子唯一标识 ──
    "atom_id":              ndarray([1,5,7,11, ...]),   # int64, PDB 原始 atom serial
    "atom_name_unique":     ndarray(["N_1","CA_1",...]),# object, 全局唯一原子名
    "auth_asym_id":         ndarray(["A","A",...]),
    "auth_atom_id":         ndarray(["N","CA",...]),
    "auth_comp_id":         ndarray(["MET","MET",...]),
    "auth_seq_id":          ndarray([1,1,...]),
    "label_asym_id":        ndarray(["A","A",...]),
    "label_atom_id":        ndarray(["N","CA",...]),
    "label_comp_id":        ndarray(["MET","MET",...]),
    "charge":               ndarray([0,0,...]),
    "hetero":               ndarray([False,False,...]),
    "ins_code":             ndarray(["","",...]),
}
```

**per-residue 展开（前 3 个残基）：**

```
[0] res_id=1  MET  token_id=0  token_position=0
    atom=N     coord=(  34.482    7.548   13.433)
    atom=CA    coord=(  35.925    7.444   13.124)
    atom=C     coord=(  36.787    7.926   14.301)
    atom=O     coord=(  37.388    8.994   14.241)

[1] res_id=2  GLY  token_id=1  token_position=1
    atom=N     coord=(  36.893    7.134   15.372)
    atom=CA    coord=(  37.574    7.480   16.635)
    atom=C     coord=(  36.662    8.179   17.655)
    atom=O     coord=(  36.802    7.987   18.859)

[2] res_id=3  ASP  token_id=2  token_position=2
    atom=N     coord=(  35.649    8.891   17.171)
    atom=CA    coord=(  34.435    9.287   17.892)
    atom=C     coord=(  33.413    8.127   18.055)
    atom=O     coord=(  32.369    8.303   18.689)
```

v3 的 `bb_input/` PDB 只含 backbone 原子（N, CA, C, O），
所以 312 residues × 4 atoms = 1248 atoms。所有 `pseudo_beta` 都走 CA fallback。

> **⚠️ TODO — v4 数据就绪后必须注意：**
>
> v4 的 `bb_input/` PDB 将是 **全原子（full-atom）** 的 monomer，包含完整侧链。
> 这意味着：
> 1. `_monomer.pt` 的 atom 数量会从 ~4N 暴增到 ~8-14N（取决于氨基酸组成）
> 2. `pseudo_beta` 将不再全部走 CA fallback——GLY 仍为 CA，其余残基可正确取 CB
> 3. `build_monomer_input_geometry` 能匹配到 GT 的侧链原子 → monomer_mask 覆盖率大幅提升
> 4. **不需要重新设计格式**：`save_atomarray_as_pt` 自动保存所有 annotation，全原子只是数组更长
> 5. **需要重新生成** `_monomer.pt`（GT 不受影响）
> 6. 文件大小会显著增加（~3-5×），需要评估磁盘空间
> 7. mesh HANDOFF #3 提到的 "KDTree 只能用 backbone 原子" 问题也会 natural fix
>
> 详见 `experiments/mesh_prior/HANDOFF20260728.md` "v4 数据就绪后" 节。

#### 为什么不能只存 CB/CA 原子？

`build_monomer_input_geometry()` 需要 **per-atom 查表**，把 monomer 坐标映射到 GT 的每个原子上：

```python
# _runtime.py: build_monomer_input_geometry()
for index in retained_indices:
    key = (chain_id, res_id, atom_name)   # ← 三元组做 key
    coordinate = lookup.get(key)           # ← 在 monomer AtomArray 里查
    if coordinate is None:
        continue                           # ← 查不到就跳过
    monomer_positions[index] = coordinate
    monomer_mask[index] = True
```

GT 里有什么原子，monomer lookup 就得有什么原子。GT 有 `("A", 1, "N")`, `("A", 1, "CA")`, `("A", 1, "C")`, `("A", 1, "O")`……

如果只存 CB/CA，则 lookup 只有 `("A", 1, "CA")` 和 `("A", 1, "CB")`。
GT 的 N、C、O 全部查不到 → `monomer_mask` 大片 False → 后续检查失败。

更致命的是最后的 **代表性原子校验**：

```python
# 每个 chain 需要 ≥4 个有限、非共线的代表性原子
representatives = [monomer_positions[matched[0]] for each residue]
if len(representatives) < 4 or matrix_rank(representatives) < 2:
    raise PhysPPIDataError("missing_mandatory_monomer_geometry")
```

只存 CA 的话，每个残基只有 1 个坐标。312 个残基 × 1 = 312 个点，**都是 CA 坐标**。
所有 CA 大致沿 backbone 排成一条线 → `matrix_rank ≈ 1` < 2 → **chain 直接被拒绝**。

因此 monomer `.pt` 必须保存 **所有 backbone 原子（N, CA, C, O）**，不能精简到只有 CB/CA。

---

### 3. `{case_id}_seq.pt`

从 chunk 级 `seq_cache.pkl` 提取的 per-sample 序列元数据。

**实际内容（以 `AF-..._seq.pt` 为例，13 KB）：**

```python
torch.load("AF-..._seq.pt", weights_only=False)

{
    "A": {
        "full_seq_1aa": "MGDDNDTDITEFILLGFSGYGFLQGHLFWGVLCIYVVTLLGNSLIVLLTLADS"
                         "ALHSPMYFFLRHFSVVEILYTTTIVPRMLADLRSSCPTIPLASCFTQLYFFA"
                         "LFGIAECCLLTAMAYDRYAAICCPLHYTTLMSQGTYTGLVGASYLAGVISGT"
                         "THSIFIFTLPFRGAKTIHHFLCDILPVLRLATASTFWGEVGNLFVTITFIF"
                         "VPFLLIVASYACILVTILGVATSQGRQKLFSTCSSHLFVVILFFGTATVAY"
                         "MRPQADSFGNTDQILTLVYTVVTPMCNPFVYSLRNKEVTGAMRRLMKRYL"
                         "WGP",                                    # str, 312 aa
        "full_seq_3aa": ["MET","GLY","ASP","ASP","ASN", ...],  # list[str], 312 entries
        "resseq_to_sequence_index": {1:0, 2:1, 3:2, ...},     # dict[int,int], PDB resseq → seq position
        "mask": [1,1,1,...],                                   # list[int], 312 entries
        "msa_eligible": True,
    },
    "B": {
        # 同上，同源二聚体序列相同
        "full_seq_1aa": "MGDDNDTDITE...",
        ...
    },
}
```

`resseq_to_sequence_index` 处理 PDB resseq 和 seq_cache position 的对齐——比如 Q-BioLiP 的 N-terminal expression tag（mesh HANDOFF 坑 #8），tag 残基在 PDB 里但不在 seq_cache 里，这个映射会跳过。

---

## 生成命令

```bash
# PhysPPI conda env 下运行：
conda activate PhysPPI

# AFDB（不需要 CCD）
python3 experiments/pdb_cache/generate_pdb_tensors.py \
    --dataset-dir /101045/jinyu/PPI/Prior_out/training/AFDB-10k-1 \
    --output-dir /101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/AFDB-10k-1 \
    --workers 64

# Q-BioLiP（需要 CCD 处理非标准残基）
python3 experiments/pdb_cache/generate_pdb_tensors.py \
    --dataset-dir /101045/jinyu/PPI/Prior_out/training/Q-BioLiP-10k-new \
    --output-dir /101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/Q-BioLiP-10k-new \
    --ccd-file /path/to/components.cif \
    --workers 64
```

## 收益估算

| 操作 | 原始耗时 | 缓存后 | 节省 |
|------|---------|--------|------|
| GT PDB 文本解析 | ~100-200ms | 0ms | ✅ |
| GT 原子标准化 | ~10-50ms | 0ms | ✅ |
| Monomer PDB 解析 × 2 | ~200-400ms | 0ms | ✅ |
| Monomer 标准化 + process × 2 | ~30-100ms | 0ms | ✅ |
| `torch.load` (GT) | — | ~5-10ms | — |
| `torch.load` × 2 (monomer) | — | ~10-20ms | — |
| `torch.load` (seq) | — | ~1-2ms | — |
| **总计** | **~350-750ms** | **~15-30ms** | **20-50×** |

## 注意事项

1. **Monomer 只有 backbone 原子**：v3 的 `bb_input/` PDB 只含 N, CA, C, O。
   所以 `pseudo_beta` 全部走 CA fallback（无 CB）。v4 全链数据到来后 natural fix。

2. **`torch.load` 需要 `weights_only=False`**：`.pt` 里混存了 tensor 和 numpy array。

3. **CCD 惰性加载**：只在遇到非标准残基时才加载 CCD 文件。

4. **`_runtime.py` 尚未修改**：adapter 接口已就绪，集成留到下一步。

## 相关文件

```
experiments/pdb_cache/
├── README.md                   ← 本文档
├── generate_pdb_tensors.py     ← 生成脚本
└── pdb_adapter.py              ← 训练时 adapter

已有缓存:
  experiments/mesh_prior/       ← KDTree mesh → residue
  experiments/residue_prior/    ← dmap / interactions / OPM

原始数据:
  /101045/jinyu/PPI/Prior_out/training/{dataset}/chunk/{chunk}/
    gt/{case_id}.pdb
    bb_input/{case_id}_{chain}.pdb
    seq_cache.pkl
```
