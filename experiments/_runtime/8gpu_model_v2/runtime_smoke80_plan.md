# 8GPU Fresh Smoke50 + Val10/Val16 + Formal Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 crop 和 pDockQ 修复后，启动一轮干净的 8GPU fresh training；`membrane_plane_loss` 和 `interaction_auxiliary_loss` 只记录日志，不参与 train/val total loss，也不参与梯度回传。

**Architecture:** 通过 loss config 增加 `include_in_total_loss` 开关，默认保持
`true` 以兼容其他实验；本次新 smoke/formal YAML 显式设为 `false`。smoke
只检查前 50 个 fresh train step；训练结束后从 `step-000050.ckpt` 单独跑固定
validation。val10 只作为快速初筛；正式 gate 是 16 个样本、8GPU、每样本 24 个
candidate PDB 的 validation artifact 检查。train50 和 val16 都通过后，才允许启动
正式 10000 step fresh run。

**Tech Stack:** PyTorch / PyTorch Lightning / OpenFold3 / PhysPPI / DDP torchrun / CSVLogger。

---

## 已确认决策

- 不从旧目录 `/101045/jinyu/PhysPPI_jinyu/PhysPPI/experiments/_runtime/8gpu_model_v2` 的 `last.ckpt` resume；旧 run 的前 59 step 已被旧 crop、旧 pDockQ 和错误 auxiliary loss 梯度污染。
- 旧失败目录原样保留，不删除、不移动、不继续写 checkpoint。
- 新建两个独立 YAML 和两个独立 output dir：
  - smoke: `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80`
  - formal: `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730`
- smoke 和 formal 都从 canonical base checkpoint fresh start：
  - `restart_checkpoint_path: /101045/jinyu/.openfold3/of3-p2-155k.pt`
  - `manual_checkpoint_loading: true`
  - `init_from_ema_weights: true`
- 默认模型配置里 `include_in_total_loss` 保持 `true`；只在本次新 YAML 里设置为 `false`。
- log-only 时用 `torch.no_grad()` 计算 membrane / interaction auxiliary loss，避免构建无用 autograd graph。
- metric 名不改：
  - `membrane_plane_loss`
  - `interaction_auxiliary_loss`
- `train/loss` 和 `val/loss` 都必须排除这两个 diagnostic-only loss。
- smoke（目录名保留历史的 `smoke80`）设置：
  - `max_steps: 50`
  - `val_check_interval: 50`
  - `limit_val_batches: 0`，fit 只检查前 50 个训练 step，不在训练进程内跑验证。
- val10 设置：
  - 从 smoke 的 `step-000050.ckpt` 手动加载 model/EMA，不执行 optimizer step，
    不恢复 scheduler 或 fit loop。
  - `mode: eval`，固定 Q-BioLiP 5 个 + AFDB 5 个，共 10 个原 val split 样本。
  - 单卡执行，避免 DDP sampler 补齐或重复样本。
  - 两个 validation dataset 的权重、crop 和 loss 配置保持原样。
- val16 设置：
  - 从同一个 `step-000050.ckpt` 手动加载 model/EMA。
  - `mode: eval`，固定 Q-BioLiP 8 个 + AFDB 8 个，共 16 个原 val split 样本。
  - 8GPU DDP 执行，验证 validation 路径在正式多卡环境下可运行。
  - `no_full_rollout_samples: 24`，每个 case 导出 24 个 candidate PDB、candidate metrics
    和 case summary。
  - artifact gate 默认开启；若实际 sample 数和预期 24 不一致，直接失败。
- formal 设置：
  - `max_steps: 10000`
  - `val_check_interval: 512`
  - 不设置 `limit_val_batches`，保持完整 validation。
- smoke train50 和独立 val16 artifact gate 都通过后才能启动 formal。
- smoke/formal 都沿用旧 YAML 的 `seed: 42` 和 `data_seed: 42`。

## 实施记录（2026-07-30）

- 两个辅助 loss 的 `include_in_total_loss` 开关、默认兼容配置和 no-grad 测试已完成。
- crop、pDockQ、loss、运行时配置等 focused tests 已通过；`7v6c_1` 真实数据加载检查通过，
  crop 后为 `768` tokens、`6133` atoms，并保留两条链。
- 第一轮带 rank 日志的 fresh smoke 跑到 step 50，50 个训练 step 连续且关键 metric
  全为有限值，独立 checkpoint 正常生成，未出现 crop geometry、pDockQ mapping/reindex、
  OOM 或 NaN 错误。
- step 35 的 `train/loss=1.015932`，同时
  `train/membrane_plane_loss=40.285519`、
  `train/interaction_auxiliary_loss=12.145834`，真实训练路径证明两个诊断项已记录但未进入
  total loss。
- 完整 validation 共 2,002 个样本，实际触发后确认不适合作为短 smoke；该尝试已停止并归档。
- 曾误启动一次新的 smoke 重跑，用户确认无需重复训练后在 step 4 主动停止；
  `_aborted_attempts/unnecessary_rerun_20260730/` 已按用户确认删除，未影响有效的
  train50 归档。
- 最终口径改为直接采用已完成的前 50 步训练结果，再从 step-50 checkpoint 单独跑
  val10 初筛和 val16 8GPU artifact gate；formal 仍跑完整 validation。
- val10 已在物理 GPU 7 上单卡完成，退出码为 `0`，总用时约 4 分 39 秒。两个
  validation loader 分别包含 Q-BioLiP 5 个和 AFDB 5 个样本；Lightning Rich
  进度条切换 loader 时重新计数，所以最终日志中的 `5/5` 是第二个 loader，不是总共
  只跑了 5 个。
- val10 结果为 `val/loss=1.091447`、
  `val/membrane_plane_loss=2.997747`、
  `val/interaction_auxiliary_loss=14.089712`。两个诊断项均被记录，但未进入
  `val/loss`。
- val10 没有 crop geometry、pDockQ mapping/reindex、OOM 或 NaN 异常，可作为快速初筛通过。
- val16 已用 8GPU DDP 完成，退出码为 `0`。实际导出 `16` 个 case 目录、`384` 个
  PDB、`16` 份 `candidate_metrics.csv`、`16` 份 `case_summary.json` 和 `8` 个
  rank timing 文件；每个 case 都有 24 个 candidate PDB。
- val16 聚合 loss 为 `val/loss=1.277732`、`val/pdockq_loss=0.213355`、
  `val/membrane_plane_loss=1.896712`、`val/interaction_auxiliary_loss=12.122858`。
  两个诊断 loss 仍被记录但未进入 `val/loss`。
- val16 结构指标行包含 `val/dockq_protein_protein_w=0.068161`、
  `val/rmsd=22.318220`；`val/model_selection=nan` 来自该小验证集上相关性统计退化，
  不是运行失败。
- val16 没有 crop geometry、pDockQ mapping/reindex、OOM、NaN 或 sample-count mismatch
  异常，train50 + val16 gate 已通过。
- validation 慢点定位：不是单卡 bug，也不是 PDB 写盘瓶颈。16 条 timing 记录显示
  full rollout 平均约 `106.664s/case`，metrics 平均约 `20.495s/case`，写 24 个 PDB
  平均约 `0.926s/case`。主成本来自 `24` 个 sample、`200` 步 diffusion 的完整 rollout；
  metrics 是第二成本。
- reader inventory 对齐时发现旧 Q-BioLiP manifest 比当前可读 `.pt` 多 29 个 ID
  （train 20、val 9）。按“不重新 split、只移除 reader 中不存在的 ID”处理后，
  active manifest、train/val ID 列表和 sampling weights 已同步；Q-BioLiP
  train/val 现为 `8685/1307`，下次训练不会再出现该 manifest warning。
- formal 第一次启动阶段暴露出新的性能问题：manifest-bound dataset 初始化时每个
  DDP rank 都在全量读取样本做 dmap 预过滤，导致长期高 CPU、GPU 未进入 forward。
  最终修复为：manifest-bound split 信任已筛好的 manifest，不做 eager dmap scan；
  非 manifest / 小样本调试路径仍保留 dmap 过滤和 warning。
- 更新后的 focused regression 为 `35 passed, 16 warnings`；smoke/val10/val16/formal
  配置契约测试包含 formal `num_sanity_val_steps: 0` 和 manifest-bound dmap scan
  行为。
- formal 曾于 `2026-07-30 18:45:32 CST` 过早 fresh 启动，8GPU DDP 已进入
  forward/backward，metrics 写到 step 6；用户明确要求“先确认验证集正确输出 PDB，
  再启动训练”，因此该尝试已停止并归档到
  `_aborted_attempts/formal_premature_before_validation_artifacts_20260730/`，不作为正式
  run。
- validation10 output 中 PDB 数量为 `0`，原因是 `validation10_eval.yaml` 未开启
  `settings.validation_artifacts`；val10 只作为早期 metrics 初筛。
- 当前正式 PDB artifact gate 是 val16：已验证 `16` 个 case、`384` 个 PDB、
  每个 case `24` 个 candidate PDB、`16` 份 `candidate_metrics.csv`、`16` 份
  `case_summary.json`；PDB 文件有 ATOM/HETATM 记录，日志无 crop geometry、
  pDockQ mapping/reindex、OOM、sample-count mismatch 或 traceback。

## 文件结构

- 修改：`openfold3/core/loss/loss_module.py`
  - 负责实现 `include_in_total_loss` 语义。
  - 保持 loss breakdown logging 不变。
  - log-only 模式下用 `torch.no_grad()` 包住对应 loss 计算。

- 修改：`openfold3/projects/of3_all_atom/config/model_config.py`
  - 给默认 loss config 增加：
    - `loss_module.membrane.include_in_total_loss: true`
    - `loss_module.interaction_auxiliary.include_in_total_loss: true`

- 新建：`experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80.yaml`
  - 从当前 `experiments/_runtime/8gpu_model_v2.yaml` 复制。
  - 改 smoke output/checkpoint 路径。
  - 改 `max_steps: 50`。
  - 改 `val_check_interval: 50`。
  - 设置 `limit_val_batches: 0`，fit 只跑训练。
  - 关闭两个 auxiliary loss 的 total-loss 贡献。

- 新建：`experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/validation10_eval.yaml`
  - `mode: eval`，手动加载归档的 `step-000050.ckpt` model/EMA。
  - 固定选择 Q-BioLiP 5 个和 AFDB 5 个 validation 样本。
  - 使用单卡，不执行训练或 optimizer step。

- 新建：`experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/validation16_eval.yaml`
  - `mode: eval`，手动加载同一个 `step-000050.ckpt` model/EMA。
  - 固定选择 Q-BioLiP 8 个和 AFDB 8 个 validation 样本。
  - 使用 8GPU DDP，`no_full_rollout_samples: 24`。
  - 开启 validation artifact 导出和 timing profile。

- 新建：`experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730.yaml`
  - 从当前 `experiments/_runtime/8gpu_model_v2.yaml` 复制。
  - 改 formal output/checkpoint 路径。
  - 保持 `max_steps: 10000`。
  - 保持 `val_check_interval: 512`。
  - 关闭两个 auxiliary loss 的 total-loss 贡献。

- 新建：`experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/RUN_NOTES.md`
  - 记录 smoke 的 git 状态摘要、测试结果、启动命令、parent log、rank logs、metrics 路径和验收结论。

- 新建：`experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730/RUN_NOTES.md`
  - 记录 formal 的 git 状态摘要、smoke gate 结果、正式启动命令、parent log、rank logs、metrics 路径和验收结论。

---

## Task 1: 给 auxiliary loss 增加 total-loss 开关

**Files:**
- Modify: `openfold3/core/loss/loss_module.py`
- Modify: `openfold3/projects/of3_all_atom/config/model_config.py`

- [ ] **Step 1: 在默认 config 里加兼容字段**

在 `openfold3/projects/of3_all_atom/config/model_config.py` 的 `loss_module` 配置中，把 membrane 和 interaction auxiliary 改成包含默认开启的开关：

```python
"membrane": {
    "normal_weight": 1.0,
    "offset_weight": 1.0,
    "include_in_total_loss": True,
},
"interaction_auxiliary": {
    "distance_weight": 1.0,
    "include_in_total_loss": True,
},
```

预期行为：没有显式覆盖 YAML 的旧实验仍保持原语义，这两个 loss 继续进入 total loss。

- [ ] **Step 2: 在 loss_module.py 里让 membrane 支持 log-only**

把当前 membrane 逻辑从“总是算完后加进 `cum_loss`”改成：

```python
membrane_cfg = self.config.get("membrane", {})
membrane_include_in_total = membrane_cfg.get("include_in_total_loss", True)

if "physppi_membrane_gt_normal" in batch:
    if membrane_include_in_total:
        l_membrane = directed_membrane_plane_loss(
            predicted_normal=output["physppi_predicted_membrane_normal"],
            predicted_offset=output["physppi_predicted_membrane_offset"],
            gt_normal=batch["physppi_membrane_gt_normal"],
            gt_offset=batch["physppi_membrane_gt_offset"],
            valid_mask=batch.get("physppi_membrane_valid"),
            config=membrane_cfg,
        )
        cum_loss = cum_loss + l_membrane
    else:
        with torch.no_grad():
            l_membrane = directed_membrane_plane_loss(
                predicted_normal=output["physppi_predicted_membrane_normal"],
                predicted_offset=output["physppi_predicted_membrane_offset"],
                gt_normal=batch["physppi_membrane_gt_normal"],
                gt_offset=batch["physppi_membrane_gt_offset"],
                valid_mask=batch.get("physppi_membrane_valid"),
                config=membrane_cfg,
            )

    losses["membrane_plane_loss"] = l_membrane.detach().clone()
```

注意：如果文件里已有局部变量名不同，保持现有风格即可，但语义必须完全一致。

- [ ] **Step 3: 在 loss_module.py 里让 interaction auxiliary 支持 log-only**

把当前 interaction auxiliary 逻辑从“总是算完后加进 `cum_loss`”改成：

```python
interaction_cfg = self.config.get("interaction_auxiliary", {})
interaction_include_in_total = interaction_cfg.get("include_in_total_loss", True)

if (
    "interaction_distance_pred" in output
    and "physppi_interaction_aux_valid" in batch
):
    if interaction_include_in_total:
        l_interaction = interaction_auxiliary_loss(
            predicted_distance=output["interaction_distance_pred"],
            target_distance=batch.get("physppi_interaction_aux_distance"),
            valid_mask=batch["physppi_interaction_aux_valid"],
            config=interaction_cfg,
        )
        cum_loss = cum_loss + l_interaction
    else:
        with torch.no_grad():
            l_interaction = interaction_auxiliary_loss(
                predicted_distance=output["interaction_distance_pred"],
                target_distance=batch.get("physppi_interaction_aux_distance"),
                valid_mask=batch["physppi_interaction_aux_valid"],
                config=interaction_cfg,
            )

    losses["interaction_auxiliary_loss"] = l_interaction.detach().clone()
```

预期行为：
- `losses["interaction_auxiliary_loss"]` 始终存在于可计算样本的 breakdown 中。
- `include_in_total_loss: false` 时，它不影响 `losses["loss"]`，也不影响 backward。

- [ ] **Step 4: 检查训练阶段加权函数没有把 breakdown 重新求和**

检查 `openfold3/projects/of3_all_atom/runner.py` 中 `apply_physppi_loss_weight()` 的调用链。

验收标准：
- 它可以缩放最终 `loss` 和 breakdown 中的显示值。
- 它不能根据 `loss_breakdown` 重新聚合一个新的 total loss。
- 如果发现它会重新聚合，改成只缩放传入的 `loss`，不要用 `membrane_plane_loss` / `interaction_auxiliary_loss` 重建 total。

---

## Task 2: 写测试证明 log-only 不反传

**Files:**
- Test: `openfold3/tests/test_loss_module.py`
- 可选 Test: `openfold3/tests/projects/of3_all_atom/physppi/model/test_runtime_loss_observability.py`

- [ ] **Step 1: 添加 membrane / interaction log-only 梯度测试**

在 loss module 测试中构造一个最小可反传场景：

```python
def test_auxiliary_losses_log_only_do_not_contribute_gradients(monkeypatch):
    import ml_collections as mlc
    import torch

    from openfold3.core.loss import loss_module as loss_module_lib
    from openfold3.core.loss.loss_module import OpenFold3Loss

    base = torch.tensor(1.0, requires_grad=True)
    membrane_pred = torch.tensor(2.0, requires_grad=True)
    interaction_pred = torch.tensor(3.0, requires_grad=True)

    def fake_confidence_loss(**kwargs):
        return base * 0.0, {}

    def fake_membrane_loss(**kwargs):
        return membrane_pred * 5.0

    def fake_interaction_loss(**kwargs):
        return interaction_pred * 7.0

    monkeypatch.setattr(loss_module_lib, "confidence_loss", fake_confidence_loss)
    monkeypatch.setattr(
        loss_module_lib,
        "directed_membrane_plane_loss",
        fake_membrane_loss,
    )
    monkeypatch.setattr(
        loss_module_lib,
        "interaction_auxiliary_loss",
        fake_interaction_loss,
    )

    config = mlc.ConfigDict(
        {
            "train_confidence_only": True,
            "confidence": {},
            "membrane": {"include_in_total_loss": False},
            "interaction_auxiliary": {"include_in_total_loss": False},
        }
    )
    batch = {
        "physppi_membrane_gt_normal": torch.zeros(1),
        "physppi_membrane_gt_offset": torch.zeros(1),
        "physppi_interaction_aux_valid": torch.ones(1, dtype=torch.bool),
    }
    output = {
        "physppi_predicted_membrane_normal": torch.zeros(1),
        "physppi_predicted_membrane_offset": torch.zeros(1),
        "interaction_distance_pred": torch.zeros(1),
    }

    loss, breakdown = OpenFold3Loss(config).loss(batch, output)
    loss.backward()

    assert "membrane_plane_loss" in breakdown
    assert "interaction_auxiliary_loss" in breakdown
    assert torch.isclose(breakdown["membrane_plane_loss"], torch.tensor(10.0))
    assert torch.isclose(
        breakdown["interaction_auxiliary_loss"],
        torch.tensor(21.0),
    )
    assert membrane_pred.grad is None
    assert interaction_pred.grad is None
```

- [ ] **Step 2: 添加默认 true 会反传的对照测试**

同一个测试文件里加对照：

```python
def test_auxiliary_losses_default_include_in_total_contributes_gradients(monkeypatch):
    import ml_collections as mlc
    import torch

    from openfold3.core.loss import loss_module as loss_module_lib
    from openfold3.core.loss.loss_module import OpenFold3Loss

    base = torch.tensor(1.0, requires_grad=True)
    membrane_pred = torch.tensor(2.0, requires_grad=True)
    interaction_pred = torch.tensor(3.0, requires_grad=True)

    def fake_confidence_loss(**kwargs):
        return base * 0.0, {}

    def fake_membrane_loss(**kwargs):
        return membrane_pred * 5.0

    def fake_interaction_loss(**kwargs):
        return interaction_pred * 7.0

    monkeypatch.setattr(loss_module_lib, "confidence_loss", fake_confidence_loss)
    monkeypatch.setattr(
        loss_module_lib,
        "directed_membrane_plane_loss",
        fake_membrane_loss,
    )
    monkeypatch.setattr(
        loss_module_lib,
        "interaction_auxiliary_loss",
        fake_interaction_loss,
    )

    config = mlc.ConfigDict(
        {
            "train_confidence_only": True,
            "confidence": {},
            "membrane": {},
            "interaction_auxiliary": {},
        }
    )
    batch = {
        "physppi_membrane_gt_normal": torch.zeros(1),
        "physppi_membrane_gt_offset": torch.zeros(1),
        "physppi_interaction_aux_valid": torch.ones(1, dtype=torch.bool),
    }
    output = {
        "physppi_predicted_membrane_normal": torch.zeros(1),
        "physppi_predicted_membrane_offset": torch.zeros(1),
        "interaction_distance_pred": torch.zeros(1),
    }

    loss, breakdown = OpenFold3Loss(config).loss(batch, output)
    loss.backward()

    assert "membrane_plane_loss" in breakdown
    assert "interaction_auxiliary_loss" in breakdown
    assert torch.isclose(breakdown["loss"], torch.tensor(31.0))
    assert torch.isclose(membrane_pred.grad, torch.tensor(5.0))
    assert torch.isclose(interaction_pred.grad, torch.tensor(7.0))
```

- [ ] **Step 3: 运行 focused loss 测试**

Run:

```bash
cd /101045/jinyu/PhysPPI_jinyu/PhysPPI
python -m pytest openfold3/tests/test_loss_module.py -q
```

Expected:

```text
passed
```

---

## Task 3: 新建 smoke50、val10 和 formal YAML

**Files:**
- Create: `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80.yaml`
- Create: `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730.yaml`

- [ ] **Step 1: 从旧 YAML 复制出 smoke YAML**

以 `experiments/_runtime/8gpu_model_v2.yaml` 为基底创建：

`experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80.yaml`

必须改这些字段：

```yaml
experiment_settings:
  output_dir: experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80
  restart_checkpoint_path: /101045/jinyu/.openfold3/of3-p2-155k.pt
  ckpt_load_settings:
    manual_checkpoint_loading: true
    init_from_ema_weights: true
    strict_loading: false

pl_trainer_args:
  max_steps: 50
  val_check_interval: 50
  limit_val_batches: 0

checkpoint_config:
  dirpath: experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/checkpoint

model_update:
  custom:
    architecture:
      loss_module:
        membrane:
          include_in_total_loss: false
        interaction_auxiliary:
          include_in_total_loss: false
```

其他训练设置沿用旧 YAML，尤其：

```yaml
experiment_settings:
  seed: 42

data_module_args:
  data_seed: 42
```

- [ ] **Step 2: 创建独立 val10 eval YAML**

从 smoke YAML 派生 `validation10_eval.yaml`：

- `mode: eval`
- `restart_checkpoint_path` 指向 smoke 的 `step-000050.ckpt`
- manual model/EMA loading：`manual_checkpoint_loading: true`
- `devices: 1`
- validation 的 `sample_ids` 固定为 Q-BioLiP 5 个 + AFDB 5 个
- 不运行 train loop，不执行 optimizer step

- [ ] **Step 2: 从旧 YAML 复制出 formal YAML**

以 `experiments/_runtime/8gpu_model_v2.yaml` 为基底创建：

`experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730.yaml`

必须改这些字段：

```yaml
experiment_settings:
  output_dir: experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730
  restart_checkpoint_path: /101045/jinyu/.openfold3/of3-p2-155k.pt
  ckpt_load_settings:
    manual_checkpoint_loading: true
    init_from_ema_weights: true
    strict_loading: false

pl_trainer_args:
  max_steps: 10000
  val_check_interval: 512

checkpoint_config:
  dirpath: experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730/checkpoint

model_update:
  custom:
    architecture:
      loss_module:
        membrane:
          include_in_total_loss: false
        interaction_auxiliary:
          include_in_total_loss: false
```

其他训练设置沿用旧 YAML，尤其：

```yaml
experiment_settings:
  seed: 42

data_module_args:
  data_seed: 42
```

- [ ] **Step 3: 验证 YAML 能被 entrypoint 解析**

Run:

```bash
cd /101045/jinyu/PhysPPI_jinyu/PhysPPI
python - <<'PY'
from pathlib import Path
from openfold3.core.config import config_utils
from openfold3.entry_points.validator import TrainingExperimentConfig

for path in [
    Path("experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80.yaml"),
    Path("experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730.yaml"),
]:
    cfg = TrainingExperimentConfig.model_validate(config_utils.load_yaml(path))
    print(path)
    print("  output_dir:", cfg.experiment_settings.output_dir)
    print("  restart_checkpoint_path:", cfg.experiment_settings.restart_checkpoint_path)
    print("  manual_checkpoint_loading:", cfg.experiment_settings.ckpt_load_settings.manual_checkpoint_loading)
    print("  max_steps:", cfg.pl_trainer_args.max_steps)
    print("  val_check_interval:", cfg.pl_trainer_args.val_check_interval)
PY
```

Expected:

```text
manual_checkpoint_loading: True
max_steps: 50
val_check_interval: 50
limit_val_batches: 0
validation10 mode: eval
validation10 total cases: 10
max_steps: 10000
val_check_interval: 512
```

---

## Task 4: 启动前回归测试和状态记录

**Files:**
- Read-only check: `experiments/_runtime/8gpu_model_v2/crop_fix_plan.md`
- Read-only check: `experiments/_runtime/8gpu_model_v2/pdockQ_fix_plan.md`
- Create: `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/RUN_NOTES.md`
- Create: `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730/RUN_NOTES.md`

- [ ] **Step 1: 记录 git 状态摘要**

Run:

```bash
cd /101045/jinyu/PhysPPI_jinyu/PhysPPI
git rev-parse HEAD
git status --short
```

把输出摘要写入两个 `RUN_NOTES.md`。不要贴完整 diff；只记录 commit、dirty 文件列表、关键修复文件列表。

- [ ] **Step 2: 跑 crop / pDockQ / loss focused tests**

Run:

```bash
cd /101045/jinyu/PhysPPI_jinyu/PhysPPI
python -m pytest \
  openfold3/tests/test_loss_module.py \
  openfold3/tests/test_pdockq_loss.py \
  openfold3/tests/projects/of3_all_atom/physppi/model/test_pdockq_training_target.py \
  -q
```

再按 `crop_fix_plan.md` 里记录的 crop focused test 命令跑 crop 回归。若命令已变化，以当前 crop 修复测试文件为准，但必须覆盖 `7v6c_1` smoke 或等价 case。

Expected:

```text
passed
```

- [ ] **Step 3: 确认没有旧训练进程**

Run:

```bash
ps -eo pid,ppid,pgid,stat,etime,%cpu,%mem,cmd | rg 'torchrun|run_openfold|CUDA_VISIBLE_DEVICES'
```

Expected:

```text
无正在运行的旧 8GPU 训练进程
```

如果存在旧训练进程，先确认它不是本次 run，再停止或换 GPU；不要让两个 8GPU run 并发抢资源。

---

## Task 5: 验收 train50 并跑独立 val10

**Files:**
- Train config: `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80.yaml`
- Eval config: `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/validation10_eval.yaml`
- Archived train50 artifacts:
  `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/_aborted_attempts/pre_val_cap_20260730/`

- [x] **Step 1: 验收已经完成的前 50 个 train step**

检查归档的 `metrics/version_1/metrics.csv`：

- step 1–50 连续，无缺失。
- `train/loss`、`train/membrane_plane_loss`、
  `train/interaction_auxiliary_loss` 都存在且为有限值。
- step 35 等非零样本证明两个诊断 loss 已记录但未加进 `train/loss`。
- rank logs 中没有 crop geometry、pDockQ mapping/reindex、OOM 或 NaN 错误。

- [x] **Step 2: 验收 step-50 checkpoint**

使用：

```text
_aborted_attempts/pre_val_cap_20260730/checkpoint/step-000050.ckpt
```

该 checkpoint 必须来自本次 fresh train50，不得指向旧 `8gpu_model_v2/last.ckpt`。

- [x] **Step 3: 单独启动 10-case validation**

在一张空闲 GPU 上运行：

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> python -m openfold3.run_openfold train \
  --runner-yaml \
  experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/validation10_eval.yaml
```

虽然 CLI 子命令名是 `train`，YAML 的 `mode: eval` 会调用
`trainer.validate()`；不会创建 train loop 或执行 optimizer step。

- [x] **Step 4: 验收 val10 metrics**

检查 `validation10_eval/metrics/version_*/metrics.csv`：

- 恰好来自固定的 Q-BioLiP 5 个 + AFDB 5 个样本。
- `val/loss`、`val/membrane_plane_loss`、
  `val/interaction_auxiliary_loss` 都存在且为有限值。
- 两个诊断 loss 只记录，不进入 `val/loss`。
- 没有 crop geometry 或 pDockQ mapping/reindex 错误。

实际结果（2026-07-30）：

```text
validation loaders: Q-BioLiP 5 + AFDB 5 = 10 cases
device: physical GPU 7, single-device validation
exit code: 0
val/loss: 1.0914468765
val/membrane_plane_loss: 2.9977467060
val/interaction_auxiliary_loss: 14.0897121429
crop / pDockQ mapping / OOM error: 0
```

单卡是有意选择：每个 loader 只有 5 个样本，8GPU validation 会让分布式 sampler
补齐重复样本；当前 PhysPPI validation dataset 不会给这些 sampler 重复项附加可靠的
`repeated_sample` 标记。Rich 进度条按当前 loader 显示，因此最终 `5/5` 代表第二个
loader 完成，不代表总样本数为 5。

---

## Task 6: smoke gate 通过后启动 formal 10000 step

**Files:**
- Use: `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730.yaml`
- Write logs under: `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730/`

- [x] **Step 1: gate 检查**

formal 启动前必须同时满足：

```text
smoke train step 1–50 连续且 loss 有限
smoke step-50 checkpoint 来自 canonical base checkpoint 的 fresh run
独立 val10 正常结束且恰好使用固定 10 个 case
train/val metrics 都有 membrane_plane_loss
train/val metrics 都有 interaction_auxiliary_loss
train50 和 val10 都没有 crop geometry error
train50 和 val10 都没有 pDockQ residue mapping/reindex error
checkpoint、train metrics 和 val10 metrics 都在 smoke 独立目录
```

把 gate 结论写入 formal `RUN_NOTES.md`。

- [ ] **Step 2: 创建 formal 日志目录**

Run:

```bash
cd /101045/jinyu/PhysPPI_jinyu/PhysPPI
mkdir -p experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730/torchrun_logs_formal
```

- [ ] **Step 3: 后台启动 formal 8GPU fresh run**

Run:

```bash
cd /101045/jinyu/PhysPPI_jinyu/PhysPPI
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  --log_dir experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730/torchrun_logs_formal \
  -m openfold3.run_openfold train \
  --runner-yaml experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730.yaml \
  > experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730/train_formal_8gpu_parent.log 2>&1 &
echo $!
```

把 PID 写入 formal `RUN_NOTES.md`。

- [ ] **Step 4: formal 早期监控**

Run:

```bash
tail -f experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730/train_formal_8gpu_parent.log
```

Expected:
- 从 canonical base checkpoint manual load，而不是从旧 `last.ckpt` native resume。
- metrics 目录新建在 formal output dir 下。
- checkpoint 写入 formal output dir 下。
- step 59 附近不再复现旧 crop geometry failure。

- [ ] **Step 5: formal 第一轮 validation 后检查 metrics**

formal `val_check_interval: 512`，所以第一次 validation 后运行：

```bash
cd /101045/jinyu/PhysPPI_jinyu/PhysPPI
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730/metrics")
metrics = sorted(root.glob("version_*/metrics.csv"))
if not metrics:
    raise SystemExit("No metrics.csv found")
path = metrics[-1]
df = pd.read_csv(path)
print(path)
for c in [
    "train/loss",
    "train/membrane_plane_loss",
    "train/interaction_auxiliary_loss",
    "val/loss",
    "val/membrane_plane_loss",
    "val/interaction_auxiliary_loss",
]:
    if c not in df.columns:
        raise SystemExit(f"Missing metric column: {c}")
    print(c, "non_null", int(df[c].notna().sum()))
PY
```

Expected:
- train 和 val 的 auxiliary component metric 都存在。
- `train/loss` / `val/loss` 表示实际优化目标，不包含 diagnostic-only auxiliary loss。

---

## Final Acceptance Criteria

- 代码层面：
  - 默认 `include_in_total_loss: true`，旧实验不受影响。
  - 新 YAML 显式设置两个 auxiliary loss 为 `include_in_total_loss: false`。
  - log-only 模式下 membrane / interaction auxiliary loss 使用 `torch.no_grad()` 计算。
  - `loss_breakdown` 仍包含两个原 metric 名。
  - `losses["loss"]` 不包含这两个 diagnostic-only 项。

- smoke train50 + val10/val16：
  - fresh start from `/101045/jinyu/.openfold3/of3-p2-155k.pt`。
  - train step 1–50 连续，`max_steps: 50`。
- 独立 val10 从 `step-000050.ckpt` 手动加载 model/EMA，不执行 optimizer step。
  - val10 固定选择 Q-BioLiP 5 个 + AFDB 5 个样本。
  - 独立 val16 从同一个 checkpoint 手动加载 model/EMA，8GPU DDP。
  - val16 固定选择 Q-BioLiP 8 个 + AFDB 8 个样本。
  - val16 每个样本输出 24 个 candidate PDB、candidate metrics 和 case summary。
  - train/val metrics 都包含 membrane 和 interaction auxiliary component。
  - 无 crop geometry error。
  - 无 pDockQ residue mapping/reindex error。
  - checkpoint 和 logs 都在 smoke 独立目录。

- formal：
  - 只有 train50 + val16 8GPU artifact gate 通过后才启动。
  - fresh start from `/101045/jinyu/.openfold3/of3-p2-155k.pt`。
  - `max_steps: 10000`。
  - `val_check_interval: 512`。
  - logs、metrics、checkpoint 都在 formal 独立目录。
  - 旧失败目录 `/101045/jinyu/PhysPPI_jinyu/PhysPPI/experiments/_runtime/8gpu_model_v2` 保持原样。

## RUN_NOTES 摘要模板

两个新目录的 `RUN_NOTES.md` 使用下面结构，内容保持摘要级，不复制完整 debug 过程：

```markdown
# 8GPU Fixed Crop/pDockQ Aux-Log Run Notes

## Purpose

- Fresh run after crop fix and pDockQ fix.
- `membrane_plane_loss` and `interaction_auxiliary_loss` are diagnostic-only.
- They are logged but excluded from train/val total loss.

## Code State

- HEAD: `<git rev-parse HEAD>`
- Dirty files summary:
  - `<git status --short output>`

## Config

- YAML: `<path>`
- Output dir: `<path>`
- Checkpoint dir: `<path>`
- Base checkpoint: `/101045/jinyu/.openfold3/of3-p2-155k.pt`
- Seed: `42`
- Data seed: `42`

## Tests

- `<command>`: PASS/FAIL
- `<command>`: PASS/FAIL

## Launch

- PID: `<pid>`
- Parent log: `<path>`
- Rank logs: `<path>`
- Metrics: `<path>`

## Result

- Smoke gate: PASS/FAIL
- Formal status: RUNNING/FAILED/COMPLETE
- Notes:
  - `<short observations>`
```
