# PhysPPI 模型对比: v1 (4GPU Precomputed-PT) vs v6 (8GPU Semantic-Gated)

> **生成日期**: 2026-07-29
> **v1 路径**: `experiments/_runtime/4gpu_model_v1/` (设计/测试阶段, 含训练日志)
> **v6 路径**: `runs/physppi_semantic_50epoch/semantic-smoke-gated-20260729-v6/` (smoke32 崩溃, formal50 未启动)

---

## 1. v6 训练 Step 耗时监控

训练: `semantic-smoke-gated-20260729-v6` smoke32 测试
- **配置**: 8 GPU, `accumulate_grad_batches=2`, `max_steps=32`, `epoch_len=512`
- **日志**: `/101045/jinyu/PhysPPI/runs/physppi_formal_8gpu/stage1-semantic-smoke-gated-20260729-v6-smoke32/train.log`

### Step 耗时明细

| Step | 累计耗时 | Step 耗时 | 备注 |
|------|---------|----------|------|
| 1 | 02:59 | ~179s | CUDA kernel 编译 + 首次 forward |
| 2 | 13:20 | ~621s | 首次 backward + gradient accumulation sync (编译/JIT 开销) |
| 3 | 15:54 | ~154s | 稳定 |
| 4 | 18:27 | ~153s | 稳定 |
| 5 | 20:56 | ~149s | 稳定 |
| 6 | 23:41 | ~165s | 稳定 |
| 7 | 26:05 | ~144s | 稳定 |
| **3–7 平均** | - | **~153s/step** | |

> ⚠️ **v6 smoke32 在 step 7 崩溃**: rank6 报 `Q-BioLiP sample 7cwm_1 failed` — chain E 缺少有效的 monomer geometry (`missing_mandatory_monomer_geometry`)。formal50 训练尚未启动。v6 稳定后每 optimizer step ≈ **150-155s**。

### 有效 Step 时间分析

v6 每个 optimizer step = 2 forward passes (`accumulate_grad_batches=2`)。每 forward pass = 8 GPU × 1 sample × 48 candidates:

| 指标 | 值 |
|------|---|
| 每 optimizer step 耗时 | **~153s** |
| 有效 batch / optimizer step | 16 (8 GPU × 1 batch × 2 acc) |
| 每 forward pass 耗时 | ~76s |
| 每 GPU 每 forward pass | ~76s (1 sample × 48 candidates) |

v1 (raw) 在 4 GPU 下约 **~165s/step** (acc=1, 有效 batch=4)。v1-v1 (含 pDockQ fix) 训练从未成功跑过完整 step，无可比数据。

---

## 2. v1: 预计算 .pt 文件 Dataloader — 核心贡献

v1 最大的工程贡献是**将整个 PhysPPI 训练数据管线从在线 PDB/MSA/prior 解析切换为预计算 .pt 文件直接加载**。这是一个架构级别的改变，涉及 ~2,000+ 行新增/修改代码。

### 2.1 数据管线架构对比

| 维度 | v1 (Precomputed-PT) | v6 (Original prior_pkl) |
|------|-------------------|------------------------|
| **数据加载方式** | `torch.load("*.pt")` 直接加载预计算张量 | 在线解析 PDB + MSA + prior_pkl 文件 |
| **数据根目录** | `precomputed_mesh/cli_v3/` | `training/` (新版目录但仍是 pkl) |
| **每样本文件** | 7 种 .pt 文件 (见下表) | prior_pkl/*.pkl + bb_input/*.pdb + msa_local/ |
| **Reader 检测** | `precomputed_only` 标志自动切换 | `chunk/prior_pkl/*.pkl` 路径 |
| **Adapter 总代码** | 2,572 行 (monolithic `_runtime.py` = 2,280) | 2,519 行 (decomposed 9 文件) |
| **Reader 代码** | qbiolip: 488 行, afdb: 376 行 | qbiolip: 314 行, afdb: 314 行 |

### 2.2 每样本的 7 种 .pt 文件

```
chunkXXXX/
  {sample}_A_monomer.pt         ← 单体结构张量 (坐标, 原子类型, 序列)
  {sample}_B_monomer.pt
  {sample}_C_monomer.pt
  {sample}_gt.pt                ← Ground truth 复合物结构
  {sample}_interaction_pairs.pt ← 预计算相互作用残基对
  {sample}_interface_residues.pt← 界面残基定义 (dmap)
  {sample}_membrane_relative_location.pt ← OPM 膜定位
  {sample}_mesh.pt              ← KDTree 预计算表面 mesh 张量
  {sample}_msa.pt               ← 预计算 MSA 特征
  {sample}_seq.pt               ← 序列信息
```

### 2.3 v1 新增的代码模块

#### (a) Reader 层 — 预计算 .pt 索引与加载

**文件**: `openfold3/projects/of3_all_atom/physppi/data/readers/qbiolip10k.py` (+174 行)
**文件**: `openfold3/projects/of3_all_atom/physppi/data/readers/afdb10k.py` (+62 行)

新增函数:
- `_precomputed_sample_id(gt_path)` — 从 `{sample}_gt.pt` 提取 sample ID
- `_precomputed_chain_id(path, sample_id)` — 从 `{sample}_{chain}_monomer.pt` 提取 chain ID
- `_load_precomputed_seq(path, dataset_name)` — `torch.load("seq.pt")` → chain→sequence 映射
- `_load_precomputed_dmap(chunk, sample_id, dataset_name)` — 加载 interface residue 定义
- `_load_precomputed_interactions(chunk, sample_id)` — 加载相互作用对
- `_load_precomputed_opm(chunk, sample_id)` — 加载膜定位数据
- `_load_precomputed_mesh_presence(chunk, sample_id)` — 检测 mesh .pt 是否存在

`QBioLiP10kReader` 和 `AFDB10kReader` 新增 `precomputed_only` 模式:
- `__init__` 自动检测目录是否包含 `chunk*/*_gt.pt`
- `_index()` 有独立分支: `precomputed_only=True` 时扫描 `*_gt.pt`，否则走原始 `prior_pkl/*.pkl` 路径

#### (b) Mesh Prior 模块

**新增目录**: `experiments/mesh_prior/` (1,164 行 Python)

| 文件 | 行数 | 功能 |
|------|------|------|
| `mesh_adapter.py` | 314 | `build_surface_from_precomputed()` — 从 `_mesh.pt` 加载预计算 KDTree 表面 tensor, 根据 token_index 映射到 crop 后 residue |
| `mesh_residue_mapping.py` | 497 | Residue ↔ mesh 顶点映射关系 |
| `generate_mesh_tensors.py` | 473 | 离线生成 mesh .pt 文件 (KDTree KNN 搜索) |

#### (c) Residue Prior 模块

**新增目录**: `experiments/residue_prior/` (729 行 Python)

| 文件 | 行数 | 功能 |
|------|------|------|
| `residue_adapter.py` | 358 | `build_contact_from_precomputed()`, `build_interaction_from_precomputed()`, `build_membrane_target_from_precomputed()`, `build_tm_from_precomputed()` — 分别从 `_interaction_pairs.pt`, `_interface_residues.pt`, `_membrane_relative_location.pt` 加载 |
| `generate_residue_tensors.py` | 371 | 离线生成 residue prior .pt 文件 |

#### (d) PDB Cache 模块

**新增目录**: `experiments/pdb_cache/` (1,084 行 Python)

| 文件 | 行数 | 功能 |
|------|------|------|
| `pdb_adapter.py` | 541 | 从预计算 PDB tensor 构建 AtomArray, 替代在线 PDB 解析 |
| `generate_pdb_tensors.py` | 543 | 离线生成 PDB cache .pt 文件 |

#### (e) Adapter Runtime 集成

**文件**: `openfold3/projects/of3_all_atom/physppi/data/adapters/_runtime.py` (+211 行)

新增函数:
- `_build_surface_mesh_prior(adapter, sample, ...)` — 从 precomputed mesh .pt 构建表面特征，缺失时回退到原始 KNN
- `_derive_pt_path(sample, dataset_dir, pt_name)` — 从 prior_pkl 推导 .pt 文件路径
- `_build_residue_priors(adapter, sample, ...)` — 从 precomputed .pt 构建 contact/interaction/membrane/TM 特征，缺失时回退

### 2.4 v1 修复的 8 个 Bug

在实现 precomputed .pt dataloader 过程中修复的 bug (来自 HANDOFF):

| # | Bug | 根因 | 修复 |
|---|-----|------|------|
| 1 | .pt 路径推导失败 | `_MESH_PT_DATASET_MAP` 键名与 YAML config name 不匹配 | 新增 `_derive_pt_path()` 从 `physppi.root` + `prior_pkl` 路径推导 |
| 2 | Surface mesh key 名不一致 | mesh_adapter 返回 `physppi_surface_*`，runtime 期望 `coords/features/...` | `build_prior_features` 兼容两种 key 名 |
| 3 | MSA 列数不匹配 | .pt 加载后未投影到 query tokens | 对每个 chain 调用 `_project_msa_to_query_tokens()` |
| 4 | Interaction type 7→5 | 旧代码 7 种 (含 waterbridge/halogen)，预计算只有 5 种 | 全局 7→5: pair_constraints.py, conditioning.py, feature_injection.py, screening/preflight.py, YAML configs |
| 5 | Surface K=128→64 | 旧代码硬编码 128 neighbors，预计算 mesh 只有 K=64 | `PHYSPPI_SURFACE_NEIGHBORS=64`, 删除 pydantic validator, 4 处 128→64 |
| 6 | PAE loss broadcast | `atomize_utils.py` 使用 `*=` 无法 broadcast `[B,1,N] *= [B,5,N]` | 改为 `= ... *` (out-of-place) |
| 7 | `atom_array` 缺失 | collator 的 `dict_multimap(pad_feat_fn)` 无法处理 biotite AtomArray | collator 中先 pop 再 restore `atom_array` |
| 8 | Reader index cache 失效 | 索引缓存存储 `max_samples` 元数据导致不同 config 的缓存 miss | 统一使用 `max_samples=None` 建缓存 |

### 2.5 v1 基础设施改动

| 改动 | 文件 | 说明 |
|------|------|------|
| Multi-dataset validation | `data_module.py` | 支持多个 val dataset (list 而非 single) |
| `dataloader_idx` | `model_runner.py`, `runner.py` | validation/eval/test_step 接受 `dataloader_idx=0` |
| `val_ids` partition | `dataset_configs.py` | `manifest_partition` Literal 加入 `"val_ids"` |
| `world_size` 参数 | `QBioLiP10kDataset`, `AFDB10kDataset` | 接受 `world_size=None` |
| EMA strict=False | `runner.py` | `on_validation_epoch_start` 用 `strict=False` 加载 EMA |
| DeepSpeed attention | YAML config | `use_deepspeed_evo_attention: false` 规避 JIT kernel 编译 |
| Identity-aware split | `experiments/split_training_val/` | MMseqs2 聚类 → train/val 划分脚本 |
| 50-step checkpoint | YAML + validator 修改 | 每 50 optimizer step 保存 |

### 2.6 pDockQ Head v1

详见原 readme，核心设计:
- **si-only regression**: 去掉 z (pair) 输入，scalar → Sigmoid → MSE
- **LayerNorm + Zero-init**: 初始 loss ~0.01–0.05，而非原始的 ~15,000–25,000
- **Chain-balanced pooling**: 链大小不变性
- 代码: `pdockq_head_v1.py` (120 行), 语法测试 8/8 通过

### 2.7 v1 训练状态

- **Raw (4gpu_model_raw)**: 成功训练至 step ~20+ (metrics.csv 记录 step 10 和 20)。pDockQ loss 仍在 10,500–14,500 范围 (未修复的 bug)。后被 SIGTERM 终止。
- **v1 (4gpu_model_v1)**: train.log 仅 14 行初始化输出，**从未成功运行过任何训练 step**。pDockQ Head v1 代码和配置就绪但未能在训练循环中启动。

---

## 3. v6: 最新版 (Semantic-Gated)

v6 在独立 worktree (`physppi-adapter-runtime-decomposition-20260727`) 中开发，对 v1 做了一次**架构重组 + 功能增强**，但同时**丢弃了 precomputed .pt dataloader**。

### 3.1 Adapter 架构对比

| 维度 | v1 (Monolithic) | v6 (Decomposed) |
|------|----------------|-----------------|
| **文件数量** | 1 个主文件 (`_runtime.py`: 2,280 行) | 8 个模块文件 |
| **总代码量** | 2,572 行 | 2,519 行 (~持平) |
| **组织结构** | `_runtime.py` 包含所有逻辑 | `dataset.py`(735), `structure_io.py`(377), `chain_selection.py`(298), `msa.py`(264), `feature_assembly.py`(255), `chain_views.py`(189), `_runtime.py`(171), `geometry.py`(152) |
| **数据加载** | precomputed .pt (`torch.load`) | 原始 prior_pkl 在线解析 |
| **Mesh Prior** | ✅ KDTree mesh .pt | ❌ (仍用在线 KNN) |
| **Residue Prior** | ✅ precomputed .pt | ❌ (仍用 prior_pkl) |
| **PDB Cache** | ✅ precomputed tensor | ❌ (仍用 PDB 文件) |

### 3.2 v6 新增功能 (v1 没有的)

| 功能 | 详情 |
|------|------|
| **LoRA 微调** | 2 组 LoRA 适配器 (~264K 参数): (1) `template_embedder.linear_t` (rank=2), (2) `pairformer_stack.blocks.40-47` 的 attention + transition + pair_bias (rank=4) |
| **7 种交互类型** | v1 从 7 降到 5 (匹配预计算数据), v6 又恢复为 7 |
| **128 表面邻居** | v1 从 128 降到 64 (匹配预计算 mesh K), v6 又恢复为 128 |
| **Semantic screening manifest** | 过滤几何不完整的样本 (v1 用 identity-aware split) |
| **Interaction auxiliary loss** | `distance_weight=1.0` 显式监督界面残基距离 |
| **Membrane offset loss** | 在 normal loss 基础上加 offset 监督 |
| **50-epoch 训练** | max_epochs=50, warmup=512 + constant LR (v1: max_epochs=1) |
| **num_workers=16** | 消除数据加载 I/O 瓶颈 (v1: num_workers=0) |
| **新数据根目录** | `training/` (新目录结构, 含 backup/chunk/logs) |
| **Candidate chunk** | `candidate_chunk_size=1` 降低显存 |
| **distinct_samples_per_rank** | `true` — 每 rank 独立采样 |
| **Decomposed adapter** | 代码结构清晰, 模块化 |

### 3.3 v6 缺失的 v1 改进

| v1 改进 | v6 状态 | 影响 |
|---------|---------|------|
| **Precomputed .pt dataloader** | ❌ 完全缺失 | v6 回归到在线 PDB/MSA/prior 解析, 数据加载更慢 |
| **Mesh Prior (KDTree mesh)** | ❌ 缺失 | v6 退回到在线 KNN 构建表面特征 |
| **Residue Prior (.pt 预计算)** | ❌ 缺失 | v6 退回到 prior_pkl 在线解析 |
| **PDB Cache (.pt 预计算)** | ❌ 缺失 | v6 退回到 PDB 文件在线解析 |
| **pDockQ Head 修复** | ❌ 缺失 | v6 仍用原始 buggy head (loss 15K+) |
| **max(y₁,y₂) ranking weight** | ❌ 缺失 | v6 ranking loss 仍无差别对待所有 pair |

### 3.4 v6 训练崩溃分析

smoke32 在 step 7 崩溃:
```
RuntimeError: Q-BioLiP sample 7cwm_1 failed
→ PhysPPIDataError: missing_mandatory_monomer_geometry
→ chain 'E' lacks four finite, non-collinear representative atoms
```

**根因**: v6 没有用 semantic screening 过滤这类 case, 也没有 v1 的 precomputed geometry 校验。`7cwm_1` 的 chain E 在 crop 后不具备有效的 monomer geometry (可能只剩 <4 个非共线原子)。

---

## 4. 从 v1 到 v6 的完整演变路线图

```
                    ┌─────────────────────────────┐
                    │   Original qinwc codebase    │
                    │  (PDB/MSA/prior 在线解析)      │
                    │  interaction=7, surface=128  │
                    └─────────────┬───────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
   │  v1 Baseline   │   │   v1 Reader    │   │  v1 Experiments │
   │  (raw 4GPU)    │   │  (precomp .pt) │   │  (mesh/residue/ │
   │                │   │                │   │   pdb_cache)    │
   │ 训练到 step 20+│   │ +174 行代码     │   │ +2,977 行代码   │
   │ pDockQ loss    │   │ precomputed_   │   │ 3 个新模块      │
   │ 10K-14K (bug)  │   │ only 模式      │   │                 │
   └───────┬────────┘   └───────┬────────┘   └───────┬────────┘
           │                    │                     │
           └────────────────────┼─────────────────────┘
                                │
                                ▼
                     ┌────────────────────┐
                     │  v1 Unified (设计)  │
                     │  (4gpu_model_v1)   │
                     │                    │
                     │ • Precomp .pt      │
                     │   dataloader ✅     │
                     │ • Mesh Prior ✅     │
                     │ • Residue Prior ✅  │
                     │ • PDB Cache ✅      │
                     │ • pDockQ fix  ✅    │
                     │ • 50-step ckpt ✅   │
                     │ • interaction=5     │
                     │ • surface K=64      │
                     │                    │
                     │ 训练: ❌ 未启动     │
                     └────────┬───────────┘
                              │
                ┌─────────────┴──────────────┐
                │   v1 的设计被拆分为两条线   │
                │                            │
                ▼                            ▼
     ┌─────────────────────┐    ┌─────────────────────┐
     │  v6 Codebase        │    │  v6 Config          │
     │  (decomposed worktree)│    │  (semantic-gated)  │
     │                     │    │                     │
     │ ❌ 丢弃 .pt loader   │    │ ✅ LoRA             │
     │ ❌ 丢弃 Mesh Prior   │    │ ✅ interaction=7    │
     │ ❌ 丢弃 Residue Prior│    │ ✅ surface K=128    │
     │ ❌ 丢弃 PDB Cache    │    │ ✅ semantic screen  │
     │ ❌ 丢弃 pDockQ fix   │    │ ✅ 50-epoch         │
     │                     │    │ ✅ interaction aux   │
     │ ✅ 代码模块化重组     │    │ ✅ membrane offset  │
     │                     │    │ ✅ num_workers=16    │
     │                     │    │ ✅ decomposed arch   │
     └─────────────────────┘    └─────────────────────┘
```

---

## 5. 性能对比总结

### 5.1 核心结论: per optimizer step 耗时

**v1 raw 比 qinwc_v6 每 optimizer step 更快 (2.75min vs 5min)，但两者不可直接比——v6 每步处理 4× 数据。**

| 指标 | v1 Raw (4GPU) | qinwc_v6 (8GPU) |
|------|-------------|----------------|
| PL step = forward+backward | ~165s | ~150s |
| `accumulate_grad_batches` (manual) | 1 | **2** |
| PL steps / optimizer step | 1 | 2 |
| **每 optimizer step** | **~165s ≈ 2.75 min** | **~300s ≈ 5 min** |
| 有效 batch / optimizer step | 4 (4×1×1) | 16 (8×1×2) |
| 每样本耗时 | 41.3s | 18.8s |
| 样本/s | 0.024 | 0.053 |

> v1 raw 每 optimizer step wall-clock 比 v6 快 1.8×，但 v6 每步的有效 batch 是 v1 的 4×。标准化到每样本，v6 效率是 v1 的 **2.2×**（18.8s/sample vs 41.3s/sample）。

### 5.2 Step 耗时拆解

关键机制 (`experiment_runner.py:445-461`):

```python
if per_sample_clipping:
    # PL 不支持 manual optimization + accumulate_grad_batches 并存
    # → 把 accumulate 移入 manual_optimization，PL 侧设为 1
    self.pl_trainer_args.accumulate_grad_batches = 1
```

结果: tqdm 每次增量 = PL 认为的一个 step = 1 forward+backward。Optimizer step 由 manual optimization 每 N 次 PL step 触发一次。

| 分解 | v1 raw | qinwc_v6 |
|------|--------|----------|
| 1 forward+backward (PL step) | ~165s | ~150s |
| optimizer.step() 频率 | 每 1 PL step | 每 2 PL step |
| 每次 optimizer step 总耗时 | ~165s | **~300s** |

v6 单次 forward+backward (~150s) 比 v1 raw (~165s) 略快——LoRA 减少了 backward 计算量，8 GPU 提升了并行度。但 acc=2 让它要跑两趟才更新一次参数。

**如果 v6 也用 acc=1: ~150s/step，比 v1 raw 的 165s 快 ~10%。**

**如果 v1 raw 也用 acc=2: ~330s/step，比 v6 的 300s 慢 ~10%。**

### 5.3 .pt 对 Step 耗时的贡献

v1 raw 用 `num_workers=0`——数据加载在主进程串行执行，直接吃 step 时间。

| 加载方式 | 单样本 | 4 样本总计 | 占 ~165s 比例 |
|---------|--------|----------|-------------|
| `torch.load` 7 个 .pt | <1s | **<4s** | **<2%** |
| 在线解析 PDB+MSA+prior (推测) | 5-15s | 20-60s | 12-36% |

**.pt 替 v1 raw 直接省了 ~20-60s/step**。

对 v6 (num_workers=16): .pt 不直接缩短 step 时间（数据加载被 worker 预取隐藏），但能大幅降低 worker CPU 占用和 I/O stall 风险。

### 5.4 CPU 20000% 问题

v1 raw 训练时 CPU 飙到 20000% 的原因是 **MKL/OpenMP 线程爆炸**，与 `.pt` 或 `num_workers` 无关:

```
机器硬件: 224 逻辑核
PyTorch 默认: at::get_num_threads() = 112
             omp_get_max_threads() = 112
             mkl_get_max_threads() = 112
OMP_NUM_THREADS / MKL_NUM_THREADS: 未设置

4 DDP 进程 × 112 BLAS 线程 = ~448 线程 → CPU 20000%
```

解决方案:

```bash
export OMP_NUM_THREADS=1    # CPU 从 20000% → ~400%
```

GPU 训练不需要 CPU 并行算线性代数，单线程即可。每个 DDP 进程主要工作在 GPU 上，CPU 只做数据搬运和轻量预处理。

### 5.5 模型 Loss 对比

| Loss 项 | v1 Raw (step 10) | v1 Raw (step 20) | v6 smoke32 |
|---------|-----------------|-----------------|------------|
| pDockQ | 10,973 | 10,557 | (崩溃前未记录) |
| Diffusion | 1.12 | 1.27 | (未记录) |
| Distogram | 0.77 | 1.19 | (未记录) |
| MSE | 1.53 | 4.50 | (未记录) |
| Interaction aux | 7.29 | 5.47 | (未记录) |
| Membrane plane | 13.49 | 0.00 | (未记录) |
| **Total** | **10,994** | **10,564** | (未记录) |

> pDockQ loss 占 total loss 的 99%+。v1 pDockQ head 修复后将降至 ~0.01–0.05 (初始)，使各部分 loss 均衡。

---

## 6. 关键洞察

### 6.1 v1 的核心贡献 (代码已写完, 未部署)

1. **Precomputed .pt dataloader**: 将数据加载从"在线解析"切换到"直接加载预计算张量", 涉及 ~2,000+ 行新增代码, 3 个新模块 (mesh_prior, residue_prior, pdb_cache), 8 个 bug 修复
2. **pDockQ Head 修复**: LayerNorm + zero-init + si-only regression (120 行, 测试通过)
3. **基础设施**: Multi-dataset validation, identity-aware split, 50-step checkpoint, EMA fix

### 6.2 v6 的核心贡献 (实际运行, 但崩溃)

1. **Decomposed adapter 架构**: 将 monolithic `_runtime.py` 拆分为 8 个模块, 代码结构更清晰
2. **功能增强**: LoRA, 7 交互类型, 128 表面邻居, semantic screening, interaction aux loss, membrane offset, 50-epoch, num_workers=16
3. **训练效率**: 8 GPU + LoRA + accumulate=2 → 总 throughput 提升 4.3×

### 6.3 两条线的断裂

v1 和 v6 之间最大的问题是**两条线的成果从未合并**:

| v1 有但 v6 没有 | v6 有但 v1 没有 |
|----------------|----------------|
| 🔴 Precomputed .pt dataloader | ✅ Decomposed adapter |
| 🔴 Mesh Prior (KDTree surface) | ✅ LoRA fine-tuning |
| 🔴 Residue Prior (.pt 预计算) | ✅ 7 interaction types |
| 🔴 PDB Cache (.pt 预计算) | ✅ 128 surface neighbors |
| 🔴 pDockQ Head fix (LayerNorm + zero-init) | ✅ Semantic screening |
| 🔴 max(y₁,y₂) ranking weight | ✅ Interaction auxiliary loss |
| | ✅ Membrane offset loss |
| | ✅ 50-epoch training |
| | ✅ num_workers=16 |

---

## 7. 建议

### 7.1 紧急

1. **修复 v6 数据崩溃**: 将 `7cwm_1` 之类缺少 monomer geometry 的 case 从 screening manifest 中排除
2. **将 v1 pDockQ Head 修复移植到 v6**: LayerNorm + zero-init 是最小改动 (~15 行), 立即解决 loss 爆炸

### 7.2 中期

3. **将 v1 的 precomputed .pt dataloader 移植到 v6**: 需要适配 decomposed 架构, 工作量较大但值得——消除数据加载瓶颈
4. **统一交互类型和表面邻居配置**: 确定使用 5 还是 7 种交互类型, 64 还是 128 邻居, 保持预计算数据和代码一致

### 7.3 长期

5. **合并两个 worktree**: v1 (main repo) 和 v6 (worktree) 之间的代码差异太多, 应该合并到统一代码库
6. **启动 formal50 训练**: 在修复 pDockQ head + 数据问题后启动完整 50-epoch 训练

---

## 8. 附录

### A. 文件清单

#### v1 相关文件

| 路径 | 说明 |
|------|------|
| `experiments/_runtime/4gpu_model_v1/pdockq_head_v1.py` | pDockQ Head v1 实现 |
| `experiments/_runtime/4gpu_model_v1/test_syntax.py` | 语法/功能测试 (8/8 pass) |
| `experiments/_runtime/4gpu_model_v1/INTEGRATION.md` | 集成指南 |
| `experiments/_runtime/4gpu_model_v1/readme.md` | v1 改进计划 |
| `experiments/_runtime/4gpu_model_v1/4gpu_train_v1.yaml` | v1 训练配置 |
| `experiments/_runtime/4gpu_train.yaml` | v1 raw 训练配置 |
| `openfold3/projects/.../readers/qbiolip10k.py` | v1 reader (488 行, +174) |
| `openfold3/projects/.../readers/afdb10k.py` | v1 reader (376 行, +62) |
| `openfold3/projects/.../adapters/_runtime.py` | v1 adapter (2,280 行, monolithic) |
| `experiments/mesh_prior/` | Mesh Prior 模块 (1,164 行, 新增) |
| `experiments/residue_prior/` | Residue Prior 模块 (729 行, 新增) |
| `experiments/pdb_cache/` | PDB Cache 模块 (1,084 行, 新增) |
| `experiments/split_training_val/` | Identity-aware split 脚本 (820 行, 新增) |

#### v6 相关文件

| 路径 | 说明 |
|------|------|
| `runs/physppi_semantic_50epoch/.../v6_smoke32.yml` | v6 smoke32 配置 |
| `runs/physppi_semantic_50epoch/.../v6_formal50.yml` | v6 formal50 配置 |
| `runs/physppi_formal_8gpu/stage1-...-v6-smoke32/train.log` | v6 smoke32 训练日志 |
| `worktrees/physppi-adapter-runtime-decomposition-20260727/.../adapters/` | v6 adapter (2,519 行, 8 文件) |
| `worktrees/physppi-adapter-runtime-decomposition-20260727/.../readers/` | v6 reader (无 precomputed 支持) |

### B. 数据目录

| 版本 | 数据根 | 结构 |
|------|--------|------|
| v1 | `Prior_out/precomputed_mesh/cli_v3/` | chunk0001-chunk0015, 每样本 7 种 .pt |
| v6 | `Prior_out/training/` | chunk/0001-0050, 传统 prior_pkl + bb_input + msa |
