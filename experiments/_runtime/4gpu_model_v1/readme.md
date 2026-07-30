# PhysPPI 4-GPU Training v1 — 优化计划

> **基线**: `experiments/_runtime/4gpu_model_raw/`（当前正在运行，**不要停止**）
> **目标**: 在 v1 中修复 pDockQ loss、优化 batch size、实现 50-step checkpoint
> **原则**: 所有修改脚本放 `4gpu_model_v1/`，不覆盖原始文件，做 A/B 对比

## 实施状态

| 阶段 | 状态 | 文件 |
|---|---|---|
| pDockQ Head 重写 | ✅ 完成，语法测试通过 | `pdockq_head_v1.py`, `test_syntax.py` |
| pDockQ Head 集成到正式代码 | ⏳ 等待用户批准 | 见 `INTEGRATION.md` |
| 50-step checkpoint | ⏳ 待实施 | - |
| 动态 N (batch 优化) | ⏳ 待实施 | - |

---

## 1. 当前状态快照 (2026-07-29)

| 指标 | 值 |
|---|---|
| 训练 step | ~17 (持续推进中) |
| 速度 | ~165 秒/step |
| pDockQ loss | **15,000–25,000** (期望 ~3.9) |
| 其他 loss | diffusion ~1.0, distogram ~0.6, MSE ~1.3 (正常) |
| checkpoint | 无 (未实现) |
| 每 step case 数 | 4 (4 rank × batch 1) |

---

## 2. pDockQ Loss 修复

### 2.1 问题诊断

当前 pDockQ loss 约 15,000–25,000，而 50-bin uniform 分布的 cross-entropy 仅为
`ln(50) ≈ 3.912`。ranking loss 被限制在 `[0, 1]`，不可能产生上万的值。

**根因**: `PDockQHead` (`openfold3/core/model/heads/pdockq.py`) 直接将 trunk 的
raw single/pair representation 输入 `nn.Linear(c_s+c_z, 50)`：
- 无 LayerNorm → trunk representation 尺度很大（数量级 10²–10³）
- 无 zero-init → 随机初始化的 linear 产生极端 logits
- softmax 后正确 bin 概率接近 0 → `-log(p) ≈ 15000+`

旧 qinwc 代码库的 `pdockq.py` 有完全相同的结构（无 LayerNorm, 无 zero-init），
所以这个 bug 是继承自旧代码的。

### 2.2 其他模型的 Loss 实现调研

#### 2.2.1 直接预测 DockQ 的模型

以下模型都直接以 DockQ（或 f_nat）为 target 做预测，与 PhysPPI 目标一致：

**A. NetTCR-struc (Deleuran & Nielsen, 2025) — 最值得借鉴**

- 论文: Frontiers in Immunology, 2025
- 代码: https://github.com/mnielLab/NetTCR-struc
- 架构: GVP-GNN (Geometric Vector Perceptron)，residue-level graph
- 输入: 二面角 + one-hot AA + chain encoding + Cα distances + 方向向量
- 输出: 直接回归 DockQ 值（0-1 连续值）

**Loss 函数 — Pair MSE（核心借鉴点）**:
```
L = MSE(y1_true, y1_pred) + MSE(y2_true, y2_pred) + MSE(Δtrue, Δpred)
其中 Δ = y1_true - y2_true, Δpred = y1_pred - y2_pred
```
- 前两项保证绝对值准确
- **第三项（pairwise difference）是关键创新**：惩罚同一 target 的两个 candidate 之间
  相对排序错误，防止模型过拟合到静态特征（如氨基酸序列）
- 训练: Adam lr=1e-4, CosineAnnealingLR, weight_decay=1e-4, batch_size=32, 100 epochs
- 效果: Spearman correlation 从 0.681 → 0.855 (+25%)

**B. PPIPoseRank (Filkin, 2025) — Ranking Loss 设计精巧**

- 论文: CTU Prague Bachelor Thesis, 2025
- 架构: PPIformer encoder + MLP scoring head
- 输出: 预测 f_nat (fraction of native contacts, 等价于 DockQ)

**Loss 函数 — Focal + Weighted Ranking（核心借鉴点）**:
```
L_focal = -α(1-ŷ)^γ * log(ŷ)          # 处理难易样本不平衡
L_rank = -[y1>y2]*log(σ(ŷ1-ŷ2))       # pairwise logistic ranking
         -[y1≤y2]*log(1-σ(ŷ1-ŷ2))
         * max(y1, y2)                  # ← 关键：用 max(y1,y2) 加权！
L_total = L_focal + β * L_ranking
```
- **`max(y1, y2)` 加权是关键创新**：让包含高 DockQ pose 的 pair 获得更大梯度权重
- 这比 PhysPPI 当前的固定 margin ranking loss 更合理——不是所有 pair 同等重要
- 效果: AUROC +0.02-0.04, Success Rate 显著提升

**C. ContrastQA (2025) — Contrastive Learning 视角**

- 论文: bioRxiv 2025.06.20.660832
- 架构: GNN + ESM-2 embeddings + KNN graph
- 输出: lDDT 分类 + 全局质量回归

**Loss 函数 — MSE + CE + DockQ-Weighted Contrastive**:
```
L_total = L_mse + 0.1*L_ce + 0.005*L_cl

L_cl = -log( exp(â·p̂/τ) / (exp(â·p̂/τ) + Σ w_i * exp(â·n̂_i/τ)) )
其中 w_i = |DockQ_anchor - DockQ_negative|  # DockQ 差值加权
```
- 同一 target 的负样本用 DockQ 差值加权，不同 target 的负样本权重为 1
- **DockQ 差值作为 contrastive weight** 是新颖思路
- 但 contrastive loss 需要大量负样本，不太适合 PhysPPI 每 step 只有 4 case × 48 candidate 的场景

**D. EquiRank (Shuvo, 2025) — Equivariant GNN**

- 代码: https://github.com/mhshuvo1/EquiRank
- 架构: EGNN ensemble
- 侧重 multimeric model quality estimation
- 在 Dockground v1 和 CAPRI 上与 DockQ 有强 Spearman correlation
- Loss 细节未公开，但证明了 equivariant GNN 对 docking quality prediction 有效

**E. DeepRank-Ab (2025) — Geometric Deep Learning**

- 论文: bioRxiv 2025.12.03.691974
- 架构: atom-level graph + Voronoi surface decomposition
- 训练: >2.3M decoys, 1442 complexes
- 效果: AF3 Top 1 success rate +35.5%, mean Top 1 DockQ >2×
- Loss 细节未公开

#### 2.2.2 AlphaFold3 本身的 Confidence Heads（对比参考）

AF3 不用 DockQ 作为预测 target，而是用：
- **pLDDT**: 50-bin CE loss（和 PhysPPI 当前结构最像）
- **PAE**: 64-bin CE loss
- **PDE**: distance error regression

AF3 的 pLDDT head 和 PhysPPI 的 pDockQ head 架构思路一致（pool → linear → CE），
但 AF3 的 head 是和整个网络一起从头训练的，不存在 random init 接 pretrained trunk
的问题。

#### 2.2.3 关键对比

| 特性 | PhysPPI 当前 | NetTCR-struc | PPIPoseRank | AF3 pLDDT |
|---|---|---|---|---|
| Target | DockQ (0-1) | DockQ (0-1) | f_nat (0-1) | LDDT (0-1) |
| Head 输出 | 50-bin logits | scalar | scalar | 50-bin logits |
| 主 Loss | CE (classification) | MSE (regression) | Focal (regression) | CE (classification) |
| Ranking | margin-based pairwise | **pairwise MSE diff term** | **max(y₁,y₂) weighted** | 无 |
| 初始化 | random (bug) | random | pretrained PPIformer | co-trained |

### 2.3 可借鉴的 Loss 设计

#### 借鉴 1: NetTCR-struc 的 Pair MSE（推荐作为方案 C）

当前 PhysPPI ranking loss 的问题：
- margin=0.05 是固定阈值，不管 DockQ 差异多大
- 只惩罚排序错误，不奖励排序正确的 pair

NetTCR-struc 的 Pair MSE 同时优化**绝对值**和**相对排序**：
```python
def pair_mse_loss(pred, target, valid):
    """MSE + pairwise difference MSE."""
    # 绝对值 loss
    l_abs = ((pred - target) ** 2 * valid).sum() / valid.sum().clamp_min(1)
    # 相对排序 loss
    diff_true = target[:, :, None] - target[:, None, :]   # [B, C, C]
    diff_pred = pred[:, :, None] - pred[:, None, :]
    l_rel = ((diff_true - diff_pred) ** 2).mean()
    return l_abs + l_rel
```

优点：
- 天然数值稳定（不会有 CE 的 log explosion）
- 不需要分 bin，避免 bin boundary 离散化误差
- 不需要 softmax 计算期望值（head 直接输出 scalar）
- Pairwise diff term 的梯度比 margin-based ranking 更平滑

缺点：
- 改动更大，需要换掉整个 head + loss
- 需要验证回归是否比分类在 DockQ 上更好

#### 借鉴 2: PPIPoseRank 的 max(y₁,y₂) 加权（推荐加入方案 A）

可以在当前 PhysPPI ranking loss 中加上 `max(y1, y2)` 权重：
```python
# 当前:
terms = torch.relu(margin - sign(diff) * pred_delta)
return terms.mean()

# 改进后:
weight = torch.max(target_left, target_right)  # 高 DockQ pair 更重要
terms = torch.relu(margin - sign(diff) * pred_delta)
return (terms * weight).sum() / weight.sum().clamp_min(1)
```

优点：
- 改动极小（几行代码）
- 让模型更关注区分"好 vs 很好"，而非"差 vs 很差"
- 与 PhysPPI 的使用场景一致（我们关心 top-K 排序质量）

#### 借鉴 3: ContrastQA 的 DockQ 差值加权（暂不采用）

Contrastive learning 需要大量负样本，PhysPPI 每 step 4 case × 48 candidate = 192 sample，
做 contrastive 太稀疏。但如果后续 batch size 增大到 8+，可以重新考虑。

### 2.4 修复方案（三个选项）

**方案 A（最小改动 — 推荐先做）**: LayerNorm + Zero-Init + max(y₁,y₂) Ranking Weight

改动点：
1. `PDockQHead`: 加 `LayerNorm` + zero-init final linear
2. `pdockq_pairwise_ranking_loss`: 加 `max(target_left, target_right)` 权重
3. 不改 loss 结构，不改 target 构造

效果：
- 初始 loss ~3.9（而非 20000）
- Ranking 关注高质量 pair
- 风险最低，改动 < 30 行

**方案 B（中等改动）**: 方案 A + 回归头替代分类头

改动点：
1. Head 输出 scalar 而非 50-bin logits
2. Loss: Pair MSE（借鉴 NetTCR-struc）
3. 保留 ChainBalancedInterfacePool

效果：
- 消除 bin boundary 离散化误差
- 数值完全稳定（MSE 不会爆炸）
- 改动 ~100 行

**方案 C（大改动 — 未来考虑）**: GNN-style regression head

- 用 GVP-GNN 或 EGNN 替换当前 Linear head
- Pool 成 scalar 直接回归 DockQ
- 需要更多训练实验验证

### 2.5 推荐策略

**v1 做方案 A**（LayerNorm + Zero-Init + max-weighted ranking）：
- 立即解决 loss 爆炸问题
- Ranking weight 改进是 low-risk high-reward
- 改动量最小，可以快速启动 v1 训练

**v1 训练过程中观察**：
- 如果 pDockQ loss 下降到 2-4 但 validation DockQ correlation 不理想
- → 切换到方案 B（Pair MSE regression）
- 如果 classification + ranking 效果很好
- → 保持方案 A

### 2.6 实施步骤

**方案 A（推荐 — 最小改动）**: LayerNorm + Zero-Init

```python
class PDockQHead(nn.Module):
    def __init__(self, c_s, c_z, no_bins=50):
        self.pool = ChainBalancedInterfacePool()
        self.layer_norm = nn.LayerNorm(c_s + c_z)
        self.linear = nn.Linear(c_s + c_z, no_bins)
        # 零初始化使初始分布为 uniform
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, si, z, asym_id, token_mask):
        pooled = self.pool(si=si, z=z, asym_id=asym_id, token_mask=token_mask)
        return self.linear(self.layer_norm(pooled))
```

效果：初始 logits 全为 0 → softmax 为 uniform → `L_dist ≈ ln(50) ≈ 3.912`。

**方案 B（备选 — 改用回归）**: 将 50-bin classification 替换为直接回归 DockQ + ranking loss，类似 PPIPoseRank 的方法。改动更大，但更接近文献主流做法。

**推荐**: 先用方案 A 快速修复，验证 loss 下降到合理范围后，再考虑是否切换到方案 B。

### 2.6 实施步骤

1. [ ] 在 `PDockQHead.__init__` 添加 `nn.LayerNorm`
2. [ ] 将 `self.linear` 权重和 bias 零初始化
3. [ ] 在 `pdockq_pairwise_ranking_loss` 加入 `max(target_left, target_right)` 权重
4. [ ] 验证基础 checkpoint（of3-p2-155k.pt）缺少新 head 时 `strict=False` 仍正常工作
5. [ ] 添加测试：验证随机输入下初始 loss ≈ ln(50)
6. [ ] 添加测试：验证 loss 有限且随训练下降
7. [ ] 记录 logits min/max/mean/std + distribution/ranking 分项用于诊断

---

## 3. Batch Size 优化

### 3.1 问题分析

当前所有 case 固定 pad 到 `N=768` token，但训练池中：
- ~53.6% 的 case 实际 token 数 ≤ 768（按 PT 权重加权）
- 短 case（如 256 token）仍然分配 768×768 的 pair 空间
- 平均 pair 元素浪费约 31.9%

### 3.2 优化方案

**阶段 A: 动态 N (batch_size=1 保持不变)**

将 `n_tokens = 768` 改为 `n_tokens = min(actual_cropped_token_count, 768)`：
- 预期 pair 元素减少 ~31.9%
- 纯 pair 路径理论加速 ~1.47×
- 端到端预估加速 ~20-35%
- **风险最低**，不改变采样/loss/optimizer 语义

**阶段 B: 长度分桶 + 自适应 batch**

基于 pair budget `C_pair = 768²` 动态调整每 GPU batch size：
- `B(N) = floor(768² / N_pad²)`
- 按 dataset 分桶避免 mixed-dataset per-case loss weight 问题
- 预期平均 local batch 从 1 提升到 ~1.8（上限 4）

**暂不实施阶段 B**，先用阶段 A 收集真实显存/耗时数据。

### 3.3 实施步骤

1. [ ] 修改 featurization 使 `n_tokens = min(actual_tokens, token_budget)`
2. [ ] 验证所有 token/pair feature 使用动态 N
3. [ ] 验证 crop 后 AtomArray、GT、mask 全对齐
4. [ ] 验证 MSA/template/PhysPPI optional prior 接受 `N < 768`
5. [ ] 记录每 case 的 actual N、atom count、MSA depth、peak memory
6. [ ] 跑 N=768 case 的一致性测试（与旧行为数值一致）

---

## 4. 50-Step Checkpoint

### 4.1 问题分析

当前 YAML 配置：
```yaml
checkpoint_config:
  every_n_steps: 1000  # ❌ 错误字段名，被 Pydantic 静默忽略
```

需要修复三个层面：
1. **Schema**: `CheckpointConfig` 只有 `every_n_train_steps`，且 `validator.py` 明确拒绝非零值
2. **目录**: 需要 `dirpath` 字段指定保存路径
3. **EMA 时序**: Lightning callback 的 `on_train_batch_end` 早于 EMA 更新，可能导致 checkpoint 中 EMA 滞后一步

### 4.2 修复方案

1. 修改 `validator.py`：允许 `every_n_train_steps > 0`
2. 添加 `dirpath` 支持到 `CheckpointConfig` 和 `experiment_runner.py`
3. 使用 `every_n_train_steps: 50`，`save_top_k: -1`（保留全部）
4. 估算磁盘：每 checkpoint ~2.5GB，50 step × ~165s = ~2.3 小时一个，10000 step ≈ 200 个 ≈ 500GB
   → 需要设置保留策略或限制数量

### 4.3 实施步骤

1. [ ] 修改 `validator.py` 放开 `every_n_train_steps` 限制
2. [ ] 在 `CheckpointConfig` 添加 `dirpath` 字段
3. [ ] 在 `experiment_runner.py` 传递 `dirpath` 到 Lightning `ModelCheckpoint`
4. [ ] 解决 EMA hook 时序问题
5. [ ] 测试：step 50 恰好保存，checkpoint 可恢复训练
6. [ ] 检查磁盘容量，设置合理的 `save_top_k`

---

## 5. 文件修改清单

### 5.1 需要修改的正式代码（改动放 v1 目录）

| 文件 | 改动内容 | v1 脚本 |
|---|---|---|
| `openfold3/core/model/heads/pdockq.py` | 加 LayerNorm + zero-init | `4gpu_model_v1/pdockq_head.patch` |
| `openfold3/entry_points/validator.py` | 放开 every_n_train_steps | 直接改（配置校验） |
| `openfold3/entry_points/experiment_runner.py` | 支持 checkpoint dirpath | 直接改（配置传递） |
| `openfold3/core/data/pipelines/sample_processing/structure.py` | 动态 N 支持 | `4gpu_model_v1/dynamic_n.patch` |
| `openfold3/core/data/pipelines/featurization/structure.py` | 动态 N 支持 | 同上 |

### 5.2 v1 配置文件

| 文件 | 说明 |
|---|---|
| `4gpu_model_v1/4gpu_train_v1.yaml` | 修改后的训练配置（checkpoint 50 step + 动态 N 开关） |
| `4gpu_model_v1/launch_4gpu_v1.sh` | 修改后的启动脚本 |
| `4gpu_model_v1/readme.md` | 本文档 |

---

## 6. 执行顺序

```
Phase 1: pDockQ 修复（最优先，阻塞训练）
  ├── TDD: 写测试 → 验证初始 loss ≈ ln(50)
  ├── 修改 pdockq.py head
  ├── 验证基础 checkpoint 加载
  └── 单卡 2-3 step smoke test

Phase 2: 50-step checkpoint
  ├── 修改 validator + experiment_runner
  ├── 处理 EMA hook 时序
  ├── 测试 step 50 保存和恢复
  └── 检查磁盘容量

Phase 3: 动态 N (batch 优化阶段 A)
  ├── 修改 featurization
  ├── 验证对齐
  ├── 单卡 A/B benchmark
  └── 4-GPU short smoke

Phase 4: 集成验证
  ├── 跑全部联合回归测试
  ├── 4-GPU short smoke (50-100 step)
  └── 确认所有 rank 日志、loss 有限、checkpoint 正常
```

---

## 7. 预期结果

| 指标 | 当前 (raw) | v1 目标 |
|---|---|---|
| pDockQ loss | 15,000–25,000 | 初始 ~3.9，随训练下降到 2-4 |
| 总 loss | 被 pDockQ 主导 | 各部分均衡 |
| checkpoint | 无 | 每 50 step 保存 |
| step 时间 | ~165s | 预估 120-135s (动态 N 后) |
| batch 效率 | 短 case 浪费 ~32% pair | 动态 N 消除浪费 |

---

## 8. 绝对不做的

- ❌ 停止/取消当前 `4gpu_model_raw` 训练
- ❌ 修改原始 `4gpu_train.yaml` 或 `launch_4gpu.sh`
- ❌ `git reset --hard` 或任何批量回退
- ❌ 在未验证前直接启动 v1 训练
- ❌ 把 AFDB loss weight 0.8 当成采样比例

---

## 9. 参考

- 当前训练 HANDOFF: `experiments/_runtime/4gpu_model_raw/HANDOFF20260729.md`
- pDockQ head: `openfold3/core/model/heads/pdockq.py`
- pDockQ loss: `openfold3/core/loss/pdockq.py`
- Loss module: `openfold3/core/loss/loss_module.py`
- Runner: `openfold3/projects/of3_all_atom/runner.py`
- Checkpoint config: `openfold3/entry_points/validator.py`, `experiment_runner.py`
- 旧代码: `/101045/jinyu/PhysPPI_jinyu/qinwc/openfold-3/`
