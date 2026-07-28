# Residue Prior 预计算

## 背景

PhysPPI 的 4 个 optional prior 中有三个是 residue-level 的：

| prior | 旧方案来源 | 独立 .pt |
|-------|-----------|----------|
| **(Non-) Interface residues** | `prior_pkl` 的 `.dmap` | `{case_id}_interface_residues.pt` |
| **Interaction pairs** | `prior_pkl` 的 `.interactions` | `{case_id}_interaction_pairs.pt` |
| **Membrane relative location** | `prior_pkl` 的 `.opm` | `{case_id}_membrane_relative_location.pt` |

旧方案训练时直接读 `prior_pkl` —— 即使只需要 dmap 也要 pickle 反序列化整个 `SimpleNamespace`（含 mesh + dmap + interactions + opm），I/O 浪费。三个 prior **分开存成独立 `.pt`**，训练时按需 load。

## 预计算产出

与 mesh .pt 放在同一目录树下：

```
/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/
  AFDB-10k-1/
    chunk0001/ ... chunk0050/
      {case_id}_mesh.pt                    ← mesh_prior
      {case_id}_interface_residues.pt      ← dmap
      {case_id}_interaction_pairs.pt       ← interactions
      {case_id}_membrane_relative_location.pt ← OPM
  Q-BioLiP-10k-new/
    chunk0001/ ... chunk0050/
      ...
```

### `{case_id}_interface_residues.pt`

来自 `.dmap`：chain 间 CA-CA 距离 ≤ 18Å 的残基对。

```python
{
    "dmap": {
        # 链对 → [r_a, r_b, distance_Å]，r_a/r_b 是 PDB resseq
        "A|B": torch.tensor([[r_a, r_b, dist], ...], dtype=torch.float32),
    },
}
```

- distance 保留用于 adapter 端 cutoff 随机增强 (8-18Å)
- 几乎所有样本都有（>99.9%）

### `{case_id}_interaction_pairs.pt`

来自 `.interactions`：5 种 PLIP 具体相互作用。

```python
{
    "interactions": {
        "A|B": {
            # [r_a, r_b, type_id]，r_a/r_b 是 PDB resseq
            # type_id: 0=hbond  1=saltbridge  2=hydrophobic
            #          3=pistacking  4=pication
            "pairs": torch.tensor([[r_a, r_b, type_id], ...], dtype=torch.int32),
            # atom names（用于从 GT PDB 实时计算质心距离做 loss）
            "atom_1": [("ND2",), ("CB", "CG"), ...],
            "atom_2": [("OE1", "OE2"), ("CD1",), ...],
        },
    },
}
```

- 不存 distance（不做模型输入），保留 atom names（loss 用）
- AFDB ~65%、Q-BioLiP ~76% 的样本有

### `{case_id}_membrane_relative_location.pt`

来自 `.opm`：OPM 膜平面 + 每链跨膜区间。

```python
{
    "res_loc": {
        # per-chain interval: [start_resseq, end_resseq, side]
        # side: 1 / -1 = 膜表面,  0 = 跨膜区
        "A": torch.tensor([[1, 22, 1], [23, 54, 0], [55, 56, -1], ...], dtype=torch.int32),
    },
    "plane": {
        "normal": torch.tensor([nx, ny, nz], dtype=torch.float32),
        "centroid": torch.tensor([cx, cy, cz], dtype=torch.float32),
        "thicken": 26.8,
    },
}
```

- `res_loc` 保留完整三态（0/1/-1），adapter 端按需取
- AFDB ~100%、Q-BioLiP ~76% 的样本有

## 设计要点

- **存 PDB resseq**：`token_index.get((chain, resseq))` 自然处理 crop，crop 掉的残基 → None → skip
- **稀疏存储**：~20-100 KB per file，比 mesh .pt（~850 KB）小一个数量级
- **三个 prior 完全独立**：哪个 .pt 存在就 load 哪个，互不影响，不存在的静默回退旧逻辑

## 生成逻辑

```
输入:
  chunk/prior_pkl/{case_id}.pkl    → .dmap, .interactions, .opm

流程:
  1. pickle.load(prior_pkl) → 取 .dmap, .interactions, .opm
  2. 分别转换为 tensor 格式
  3. 三个 .pt 独立保存

输出:
  {case_id}_interface_residues.pt        (dmap 非空则生成)
  {case_id}_interaction_pairs.pt         (interactions 非空则生成)
  {case_id}_membrane_relative_location.pt (opm 非空则生成)
```

不需要 seq_cache 对齐、不需要 KDTree、不需要坐标变换。本质上是 `prior_pkl → .pt` 的格式转换。

## 训练时 adapter 接口

### Interface residues

```python
def build_contact_from_precomputed(pt_path, token_index, cutoff):
    """对上 build_contact_features(sample.dmap, token_index, cutoff)"""
    pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    pair_mask = torch.zeros(N, N, dtype=torch.bool)
    for chain_pair, edges in pt["dmap"].items():
        left_ch, right_ch = chain_pair.split("|")
        for r_a, r_b, dist in edges.tolist():
            if dist > cutoff:
                continue
            ta = token_index.get((left_ch, int(r_a)))
            tb = token_index.get((right_ch, int(r_b)))
            if ta is None or tb is None:
                continue
            pair_mask[ta, tb] = True
            pair_mask[tb, ta] = True
    return {"pair_mask": pair_mask, "available": True, "valid": bool(pair_mask.any()), "cutoff": cutoff}
```

### Interaction pairs

```python
def build_interaction_from_precomputed(pt_path, token_index):
    """对上 build_interaction_features(sample.interactions, token_index)"""
    pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    type_mask = torch.zeros(N, N, 5, dtype=torch.bool)
    aux = []
    for chain_pair, data in pt["interactions"].items():
        left_ch, right_ch = chain_pair.split("|")
        pairs = data["pairs"].tolist()
        for i, (r_a, r_b, type_id) in enumerate(pairs):
            ta = token_index.get((left_ch, int(r_a)))
            tb = token_index.get((right_ch, int(r_b)))
            if ta is None or tb is None:
                continue
            type_mask[ta, tb, int(type_id)] = True
            type_mask[tb, ta, int(type_id)] = True
            aux.append((ta, tb, int(type_id), data["atom_1"][i], data["atom_2"][i]))
    return {"type_mask": type_mask, "available": True, "valid": bool(type_mask.any()), "aux": tuple(aux)}
```

### Membrane relative location

```python
def build_tm_from_precomputed(pt_path, token_index):
    """对上 build_tm_features(res_loc, token_index) → [N] bool mask"""
    pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    tm_mask = torch.zeros(N, dtype=torch.bool)
    for chain_id, intervals in pt["res_loc"].items():
        for start, end, side in intervals.tolist():
            if int(side) != 0:   # 只要跨膜区 (side==0)
                continue
            for r in range(int(start), int(end) + 1):
                t = token_index.get((chain_id, r))
                if t is not None:
                    tm_mask[t] = True
    return tm_mask

def build_membrane_target_from_precomputed(pt_path):
    """对上 build_membrane_target(opm, is_membrane)"""
    pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    p = pt["plane"]
    return {"valid": True, "normal": p["normal"], "point": p["centroid"]}
```

## 集成方式（参照 mesh_prior）

已在 `_runtime.py` 中完成集成：

- **新增 import**：`_build_contact_from_pt`, `_build_interaction_from_pt`, `_build_tm_from_pt`, `_build_membrane_target_from_pt`
- **新增 `_derive_pt_path()`**：从 `prior_pkl` 路径推导预计算 `.pt` 路径（与 mesh .pt 同目录）
- **新增 `_build_residue_priors()`**：统一加载 3 个 .pt，每个独立存在则返回，不存在则返回 None
- **修改 `build_prior_features()`**：contact / interaction / tm / membrane 四个 prior 优先用预计算，`.pt` 不存在或加载异常静默回退旧逻辑

修改的文件：
```
openfold3/projects/of3_all_atom/physppi/data/adapters/_runtime.py
```

## 预计算结果汇总

| | AFDB | Q-BioLiP |
|---|---|---|
| `_mesh.pt` | 10,000 (100%) | ~9,910 (99.1%) |
| `_interface_residues.pt` | 10,000 (100%) | 9,992 (99.9%) |
| `_interaction_pairs.pt` | 6,562 (65.6%) | 9,712 (97.1%) |
| `_membrane_relative_location.pt` | 9,758 (97.6%) | 7,546 (75.5%) |
| 大小 | ~8 GB | ~7 GB |

> mesh .pt 仍在生成中（64 workers），上表为预期总数。

## 与 mesh_prior 的对比

| | mesh_prior | interface_residues | interaction_pairs | membrane_relative_location |
|---|---|---|---|---|
| 预计算脚本 | `generate_mesh_tensors.py` | `generate_residue_tensors.py` | 同左 | 同左 |
| 输入 | prior_pkl + bb_input + seq_cache + GT PDB | prior_pkl | prior_pkl | prior_pkl |
| 核心算法 | cKDTree + resseq 对齐 | pickle → tensor | pickle → tensor | pickle → tensor |
| 输出 | `{case_id}_mesh.pt` (~850 KB) | `{case_id}_interface_residues.pt` (~20-100 KB) | `{case_id}_interaction_pairs.pt` (~5-20 KB) | `{case_id}_membrane_relative_location.pt` (~2-5 KB) |
| 增强 | 坐标变换 | cutoff ∈ [8, 18]Å | 无 | 无 |
| fallback | `build_surface_features()` | `build_contact_features()` | `build_interaction_features()` | `build_tm_features()` + `build_membrane_target()` |

## 生成命令

```bash
# AFDB
python3 experiments/residue_prior/generate_residue_tensors.py \
    --dataset-dir /101045/jinyu/PPI/Prior_out/training/AFDB-10k-1 \
    --output-dir /101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/AFDB-10k-1 \
    --workers 64

# Q-BioLiP
python3 experiments/residue_prior/generate_residue_tensors.py \
    --dataset-dir /101045/jinyu/PPI/Prior_out/training/Q-BioLiP-10k-new \
    --output-dir /101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/Q-BioLiP-10k-new \
    --workers 64
```

## 文件

| 文件 | 说明 |
|------|------|
| `README.md` | 本文档 |
| `generate_residue_tensors.py` | 预计算脚本：prior_pkl → 三个独立 `.pt` |
| `residue_adapter.py` | 训练时 adapter：`torch.load(.pt)` → contact / interaction / membrane features |
