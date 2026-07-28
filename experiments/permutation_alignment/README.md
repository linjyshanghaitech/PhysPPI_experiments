# 置换对齐实验

## 这个目录是干什么的？

训练 PhysPPI 时，模型输出的 atom positions 和 ground truth (GT) 的 atom positions
**排列顺序不一定一致**。如果不做对齐就直接算 loss，会把原本正确的预测判为错误。

这个目录下的脚本用于**审计和验证置换对齐的正确性**。

---

## 为什么 dataset 已有 chain ID 匹配，还需要 alignment？

这是两个不同的阶段：

| 阶段 | 做什么 | 发生在 |
|---|---|---|
| **Chain 选择** | 从多链复合物中选哪些链参与训练 | 数据预处理（训练前） |
| **Atom 置换对齐** | 重排 GT atom 顺序使其与 prediction 一致 | loss 计算（训练中） |

举例：prediction 的 atom 顺序是 `[A链第3残基, A链第1残基, B链...]`，
GT 的 atom 顺序是 `[A链第1残基, A链第3残基, B链...]`。两边 chain ID 都是对的，
但 atom 行的排列不同。不重排 GT 直接算 loss 会得到无意义的误差。

PhysPPI 中 pred 和 GT 共享同一结构源的 `token_index`——这意味可以**直接按 token_index 对齐**，不需要复杂的链匹配算法。这是本次修复的核心洞察。

---

## 旧对齐逻辑（修复前）

```
safe_multi_chain_permutation_alignment()
  │
  ├── protected (优先)
  │     └── find_protected_mol_permutation()
  │           基于 mol_entity_id / mol_sym_id / mol_sym_component_id
  │           用 Hungarian 算法做 pred↔GT 链级别组合匹配
  │           匹配后做 Kabsch 刚体对齐 + 原子对称性解析
  │
  └── identity (兜底，protected 失败时)
        └── find_sequence_compatible_identity_mol_permutation()
              序列兼容的恒等映射，精度最低但不会失败
```

**关键问题**：

1. `asym_id` / `entity_id` / `sym_id` 在 `structure.py` 中被放在 `if not is_gt:` 分支内 → **GT 侧缺失这三个特征**。`protected` 匹配只能依赖 `mol_entity_id` / `mol_sym_id` 等分子属性，无法用更直观的链标签校验。

2. 没有利用 `token_index`。PhysPPI 中 pred 和 GT 共享同一个 token_index，天然可以直连，但旧代码跳过了这条路。

3. token 中心原子提取用的是 `get_token_center_atoms()`，先提取全部再 mask padding，可能索引到无效的 padding 位置。

---

## 新对齐逻辑（修复后）

```
safe_multi_chain_permutation_alignment()
  │
  ├── 🆕 token_identity (最优先)
  │     └── get_token_identity_gt_token_subset_index()
  │           pred.token_index ↔ gt.token_index 直连
  │           建立 token 级 1:1 映射
  │           → 用 asym_id 做一致性校验（映射后不对称则拒绝）
  │         → Kabsch 刚体对齐
  │         → get_token_identity_atom_permutation_index()
  │           token 内部 atom 按原始顺序对应
  │           对称原子组按 RMSD 最小化选最优置换
  │
  ├── protected (次选，token_identity 失败时)
  │     └── 同旧逻辑
  │
  └── identity (最后兜底)
        └── 同旧逻辑
```

**修改要点**：

1. **`structure.py`**：`asym_id` / `entity_id` / `sym_id` 提升到 `if not is_gt:` 外侧 → pred 和 GT 双侧都有。

2. **`permutation_alignment.py`**：
   - 新增 `chain_assignment="token_identity"` 路径，放在最优先位置
   - token 映射用 `token_index` 直连 + `asym_id` 校验
   - atom 映射在 token 内部按原始顺序，对称原子通过 `ref_space_uid` + RMSD 解析
   - token 中心提取改用 `_get_valid_token_center_atoms()`——先过滤 padding 再提取，处理单原子 token
   - 回退链变为 `token_identity → protected → identity`

---

## 三种对齐策略对比

| 策略 | 原理 | token 映射方式 | chain 信息 | 何时失败 |
|---|---|---|---|---|
| **`token_identity`** 🆕 | pred/GT 按 `token_index` 直连 | `pred.token_index[i] → GT第几行` | `asym_id` 做映射后校验 | token_index 重复/缺失/asym_id 不一致 |
| **`protected`** 旧 | 分子属性做组合优化 | Hungarian 匹配 pred↔GT 链 | `mol_entity_id`/`mol_sym_id` 驱动匹配 | 分子属性不一致 |
| **`identity`** 旧 | 序列兼容恒等映射 | 按序列相似度贪心匹配 | 不使用 | 几乎不会失败 |

**为什么 `token_identity` 在 PhysPPI 中基本总成功？** 因为 pred 和 GT 来自同一结构源，token_index 天然一致且唯一。不需要复杂的 Hungarian 链匹配——直接按 token_index 查表就行。

**为什么还保留 `protected` 和 `identity`？** 作为安全网。如果未来数据预处理有变（不同的 tokenization、augmentation），token_index 可能不再 1:1 匹配，需要下层策略接管。

---

## Atom 级别对齐细节

`token_identity` 路径的 atom 对齐分两步：

1. **token 内部顺序映射**（`_token_identity_atom_index_before_symmetry`）：token 对齐后，每个 pred token 的 N 个 atom → GT 对应 token 的 N 个 atom，按原始顺序一一对应。因为 pred 和 GT 来自同一结构，atom 数量和顺序天然一致。

2. **对称原子解析**（`get_token_identity_atom_permutation_index`）：同一 `ref_space_uid` 的原子组（如 Phe 的 CD1/CD2、Asp 的 OD1/OD2），尝试所有预计算的对称置换，选 RMSD 最小的那个。

这回答了你之前的问题：**atom 对齐就是按 index 对齐的**，只不过 token 级别按 `token_index`，token 内部按 atom 序号，对称原子额外做最优置换选择。

---

## 代码新旧分类

### 旧代码（从其他目录重构迁移，无逻辑改动）

| 文件 | 用途 |
|---|---|
| `build_minimal_alignment_arrays.py` | 从 ColabFold MSA 缓存构建对齐数组 |
| `audit_v1_v32_data_errors.py` | 扫描 20k 样本的 PDB 数据异常 |
| `audit_protected_chain_matching.py` | 重放 protected chain matching 并审计 |

### 新代码（`01eb946` 修复引入）

| 文件 | 用途 |
|---|---|
| `inspect_chain_identity_alignment.py` | 对单个样本输出 token 级别对齐诊断报告 |

---

## `asym_id` / `entity_id` / `sym_id` 是否已加入流程？✅

| 标签 | 特征化 (pred+GT) | 参与对齐 | 参与诊断 |
|---|---|---|---|
| `asym_id` | ✅ 双侧计算 | ✅ 映射后一致性校验 | ✅ mismatch 计数 |
| `entity_id` | ✅ 双侧计算 | —（存在于特征中） | ✅ mismatch 计数 |
| `sym_id` | ✅ 双侧计算 | —（存在于特征中） | ✅ mismatch 计数 |

---

## 修改文件清单（汇报用）

### 核心库

| 文件 | 改动 |
|---|---|
| `openfold3/core/data/pipelines/featurization/structure.py` | `asym_id`/`entity_id`/`sym_id` 对 pred+GT 双侧计算 |
| `openfold3/core/utils/permutation_alignment.py` | 新增 `token_identity` 对齐通路；回退链调整 (+395/-35 行) |

### 实验

| 文件 | 改动 |
|---|---|
| `experiments/permutation_alignment/inspect_chain_identity_alignment.py` | 新增：token 对齐诊断脚本 |
| `configs/experiments/permutation_alignment/inspect_chain_identity_alignment.toml` | 新增：诊断实验配置 |

### 测试

| 文件 | 改动 |
|---|---|
| `openfold3/tests/core/utils/test_permutation_alignment.py` | 新增 6 个 token_identity 路径测试 |
| `openfold3/tests/.../test_structure.py` | 验证 GT 侧链标签存在性 |
| `tests/scripts/physppi/test_experiment_config.py` | 注册新实验配置 |
| `tests/experiments/permutation_alignment/test_inspect_chain_identity_alignment.py` | 新增：诊断脚本集成测试 |

---

## 运行方式

```bash
python scripts/physppi/submit_experiment.py \
    --config configs/experiments/permutation_alignment/<name>.toml
```

输出路径：`${PHYSPPI_OUTPUT_ROOT}/experiments/permutation_alignment/`

---

## 提交历史

```
01eb946 fix: prefer token identity alignment for PhysPPI   ← 🔧 Bug 修复
2467a2d fix: align experiment configuration contracts
1f9268a feat: add reproducible permutation alignment experiments
18d9d57 fix: clean portable audit resources
7a1535a refactor: isolate protected chain matching audit     ← 旧代码迁移
fc123ff refactor: isolate v1-v32 data audit experiment        ← 旧代码迁移
41d4319 refactor: isolate alignment array experiment          ← 旧代码迁移
```

---

## 结论

### 相对于官方 openfold3 的差异

本项目的 `openfold3/` 核心库基于官方 openfold3 修改。
官方版本可 clone 到以下路径做 diff 对比：

```bash
git clone https://github.com/aqlaboratory/openfold3.git /101045/jinyu/PhysPPI_jinyu/official
```

| 文件 | 修改者 | 差异概述 |
|---|---|---|
| `openfold3/.../structure.py` | **Jinyu Lin** | `asym_id`/`entity_id`/`sym_id` 提升到 `if not is_gt:` 外侧 |
| `openfold3/.../permutation_alignment.py` | **Jinyu Lin** | 新增 `token_identity` 对齐通路；回退链调整 |
| `openfold3/tests/.../test_permutation_alignment.py` | **Jinyu Lin** | 新增 6 个 token_identity 路径测试 |
| `openfold3/tests/.../test_structure.py` | **Jinyu Lin** | 验证 GT 侧链标签存在性 |

> 所有被修改的 `.py` 文件均已添加修改头标注（文件头 `# Modified by Jinyu Lin ...`）和
> 行内标注（`# PhysPPI modification (Jinyu Lin, ShanghaiTech): ...`），
> 方便用 `grep -rn "Jinyu Lin" openfold3/` 快速定位所有差异点。

### 修复了什么

**一句话**：`structure.py` 中 `asym_id` / `entity_id` / `sym_id` 的计算原本在
`if not is_gt:` 分支内（仅预测侧有），修复后**提升到分支外侧**（预测侧和 GT 侧都有），
同时 `permutation_alignment.py` 新增 `token_identity` 对齐策略利用这些链标签做校验。

### 正式代码改动位置

**文件**：`openfold3/core/data/pipelines/featurization/structure.py`

| 改动位置 | 说明 | 标注 |
|---|---|---|
| 文件头（第 15–27 行） | 新增 Jinyu Lin 修改声明 | ✅ 已标注 |
| 第 157–159 行 | `asym_id`/`entity_id`/`sym_id` 计算，从 `if not is_gt:` 内部提升到外部 | ✅ 行内标注 |

```python
# structure.py 第 157–180 行（修复后）
# UPSTREAM: asym_id/entity_id/sym_id 在 if not is_gt: 内部
# PhysPPI:  提升到 if not is_gt: 之前，使 GT 侧也能获得这些特征

    # PhysPPI modification (Jinyu Lin, ShanghaiTech):
    #   Chain labels are present on both target and GT AtomArrays...
    chain_ids_token = atom_array.chain_id[token_starts]
    ...
    features["asym_id"] = asym_id          # ← 现在 pred 和 GT 都有
    features["entity_id"] = torch.tensor(...)
    features["sym_id"] = torch.tensor(...)

    if not is_gt:                           # ← 第 180 行，原本含以上代码，已移出
        ...
```

**文件**：`openfold3/core/utils/permutation_alignment.py`

| 改动位置 | 说明 | 标注 |
|---|---|---|
| 文件头（第 15–34 行） | 新增 Jinyu Lin 修改声明 | ✅ 已标注 |
| 新增 `get_token_identity_gt_token_subset_index()` | token_index 直连映射 + asym_id 校验 | ✅ 函数头标注 |
| 新增 `get_token_identity_atom_permutation_index()` | token→atom 级联置换 + 对称解析 | — 内部函数 |
| 新增 `_alignment_token_center_atom_indices()` | 安全的 token 中心索引（处理单原子/padding） | — 内部函数 |
| 新增 `_get_valid_token_center_atoms()` | 先过滤 padding 再提取 token 中心 | — 内部函数 |
| `single_batch_multi_chain_permutation_alignment()` 中新增 `token_identity` 分支 | 新增第三种 chain_assignment 策略 | ✅ 行内标注 |
| `safe_multi_chain_permutation_alignment()` 回退链变更 | `protected→identity` 改为 `token_identity→protected→identity` | ✅ 行内标注 |

### 快速定位所有修改点

```bash
# 在项目根目录执行，列出所有 Jinyu Lin 的修改标注
grep -rn "Jinyu Lin" openfold3/

# 与官方版本做完整 diff
diff -r /101045/jinyu/PhysPPI_jinyu/official/openfold3/ \
        /101045/jinyu/PhysPPI_jinyu/PhysPPI/openfold3/
```

### 涉及的全部文件（绝对路径）

```
/101045/jinyu/PhysPPI_jinyu/PhysPPI/openfold3/core/data/pipelines/featurization/structure.py
/101045/jinyu/PhysPPI_jinyu/PhysPPI/openfold3/core/utils/permutation_alignment.py
/101045/jinyu/PhysPPI_jinyu/PhysPPI/openfold3/tests/core/utils/test_permutation_alignment.py
/101045/jinyu/PhysPPI_jinyu/PhysPPI/openfold3/tests/core/data/pipelines/featurization/test_structure.py
/101045/jinyu/PhysPPI_jinyu/PhysPPI/experiments/permutation_alignment/inspect_chain_identity_alignment.py
/101045/jinyu/PhysPPI_jinyu/PhysPPI/configs/experiments/permutation_alignment/inspect_chain_identity_alignment.toml
/101045/jinyu/PhysPPI_jinyu/PhysPPI/tests/scripts/physppi/test_experiment_config.py
/101045/jinyu/PhysPPI_jinyu/PhysPPI/tests/experiments/permutation_alignment/test_inspect_chain_identity_alignment.py
```

### GitHub 推送状态

| 项目 | 状态 |
|---|---|
| 远程仓库 | `git@github.com:linjyshanghaitech/PhysPPI_experiments.git` |
| 修复 commit | `01eb946` |
| 分支 | `main` |
| 已推送至 GitHub | ✅ 是（`origin/main` 已同步至 `01eb946`） |
| 本文档 + 代码标注 | 本地修改中，待 commit + push |
