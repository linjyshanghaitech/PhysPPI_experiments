# 8GPU Smoke Train50 + Val10/Val16 运行摘要

## 目的

- 在 crop 和 pDockQ 修复后进行 fresh 8GPU smoke。
- `membrane_plane_loss` 和 `interaction_auxiliary_loss` 只计算并记录，不进入 train/val total loss，也不参与梯度回传。
- smoke 只验收前 50 个训练 step；随后从 step-50 checkpoint 单独跑 validation。
  val10 是快速初筛；最终 formal gate 是 16 个样本、8GPU、每样本 24 个 candidate
  PDB 的 validation artifact 检查。

## 代码状态

- 记录时间：`2026-07-30 12:48:44 CST`
- HEAD：`96a522a0237591fd45ac515e57a5926b7700d079`
- worktree：dirty，`44` 个已跟踪文件修改，`20` 个未跟踪条目。
- 本次相关关键文件：
  - `openfold3/core/loss/loss_module.py`
  - `openfold3/projects/of3_all_atom/config/model_config.py`
  - `openfold3/tests/test_loss_module.py`
  - `openfold3/core/loss/pdockq.py`
  - `openfold3/projects/of3_all_atom/runner.py`
  - `openfold3/projects/of3_all_atom/physppi/data/adapters/_runtime.py`
  - `openfold3/projects/of3_all_atom/physppi/data/adapters/chain_views.py`
  - crop/pDockQ 对应 focused tests
- 旧失败目录 `experiments/_runtime/8gpu_model_v2` 保持原样。

## 配置

- YAML：`experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80.yaml`
- output：`experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80`
- checkpoint：`experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/checkpoint`
- base checkpoint：`/101045/jinyu/.openfold3/of3-p2-155k.pt`
- fresh load：`manual_checkpoint_loading=true`，`init_from_ema_weights=true`
- seed：`42`
- data seed：`42`
- max steps：`50`
- validation interval：`50`
- fit validation batch 上限：`0`（训练进程只跑 50 个 train step）
- val10 YAML：
  `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/validation10_eval.yaml`
- val10：从归档的 `step-000050.ckpt` 手动加载 model/EMA，不恢复训练状态；
  固定 Q-BioLiP 5 个 + AFDB 5 个样本
- val16 YAML：
  `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/validation16_eval.yaml`
- val16：从同一个 `step-000050.ckpt` 手动加载 model/EMA，8GPU DDP，固定
  Q-BioLiP 8 个 + AFDB 8 个样本，每个样本输出 24 个 candidate PDB

## 启动前验证

- loss + pDockQ focused tests：`24 passed`。
- crop + Q-BioLiP + AFDB focused tests：`65 passed`。
- loss weighting + runtime observability tests：`5 passed`。
- `7v6c_1` 真实 dataset smoke：PASS，`768` tokens、`6133` atoms，保留两条链（`asym_id=[1, 2]`）。
- YAML schema 和 merged model config：PASS。
- merged config 中两个 `include_in_total_loss`：`false`。
- `py_compile`、`git diff --check` 和 trailing-whitespace 检查：PASS。
- smoke/val10/val16/formal runtime config 契约测试：`4 passed`。

## 已归档尝试

- attempt 1：仅完成初始化，未产生 step/metrics/checkpoint；因 rank log 参数缺失主动停止。
- attempt 2：fresh 跑到 step 50，生成 `step-000050.ckpt` 和 `last.ckpt`；
  50 个 train step 连续且关键 metric 全为有限值，crop/pDockQ/OOM/NaN 错误命中为 0。
- attempt 2 的 step 35：
  `train/loss=1.015932`、`train/membrane_plane_loss=40.285519`、
  `train/interaction_auxiliary_loss=12.145834`，证明两个诊断 loss 都已记录但未计入 total。
- attempt 2 在完整 validation 上运行过慢：验证集共 2,002 个样本，等待约 6 分钟仍未完成，
  按每 rank 约 250 batch 估算不适合作为 smoke gate，因此主动停止。
- 上述产物归档在：
  `_aborted_attempts/pre_val_cap_20260730/`。
- 用户确认 smoke 只需要前 50 步训练结果，因此直接采用 attempt 2 的 train50 和 checkpoint，
  不再重复训练。
- 曾误启动一次新的重跑；用户纠正后在 step 4 停止。目录
  `_aborted_attempts/unnecessary_rerun_20260730/` 已按用户确认删除，
  有效的 `pre_val_cap_20260730` train50 归档未受影响。
- 最终配置为 smoke train50 + 独立 val10 初筛 + 独立 val16 8GPU artifact gate；
  formal 保持完整 validation。
- 更新后的 smoke/val10/val16/formal 配置契约测试：`4 passed`。

## Validation10 结果

- train50：`PASS`
- train metrics：
  `_aborted_attempts/pre_val_cap_20260730/metrics/version_1/metrics.csv`
- step-50 checkpoint：
  `_aborted_attempts/pre_val_cap_20260730/checkpoint/step-000050.ckpt`
- val10：`PASS`
- val10 第一次启动仍初始化完整 train manifest，确认多余后主动停止，日志归档在
  `validation10_eval/_aborted_attempts/full_train_manifest_init/`。
- eval YAML 的两个 train dataset 已缩为各 1 个 registry 占位 ID；不会运行 train batch。
- val10 第二次启动触发 PyTorch 2.6 native `weights_only=True` 限制；改用项目 manual
  checkpoint loader，仅加载可信本地 checkpoint 的 model/EMA，不恢复 optimizer 或 fit loop。
- val10 启动时间：`2026-07-30 17:06:00 CST`
- val10 结束时间：`2026-07-30 17:17:55 CST`
- val10 退出码：`0`
- val10 设备：物理 GPU 7，单卡
- val10 output：
  `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/validation10_eval/`
- val10 parent log：
  `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/validation10_eval/validation10_parent.log`
- val10 metrics：
  `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/validation10_eval/metrics/version_0/metrics.csv`
- val10 没有开启 `settings.validation_artifacts`，因此没有导出 PDB、candidate metrics
  或 case summary；该目录中 PDB 数量为 `0` 是配置预期。
- val10 metrics 中没有 `val/pdockq_loss`，说明它是早期快速初筛结果，不作为最终
  pDockQ/artifact gate。
- 两个 validation loader 分别为 Q-BioLiP 5 个和 AFDB 5 个，共 10 个固定样本。
  Lightning Rich 进度条在 loader 切换时重置，最终 `5/5` 是第二个 loader 的显示。
- 单卡避免 5-case loader 在 8GPU DistributedSampler 下被补齐成重复样本并污染均值。

val10 聚合值：

| metric | value |
|---|---:|
| `val/loss` | 1.0914468765 |
| `val/membrane_plane_loss` | 2.9977467060 |
| `val/interaction_auxiliary_loss` | 14.0897121429 |

`val/loss` 明显小于两个诊断项，且实现/单测保证
`include_in_total_loss: false`，因此两个 auxiliary loss 只记录，不进入 val total。
日志中没有 crop geometry、pDockQ mapping/reindex、OOM 或 NaN 异常。最终应以
Validation16 8GPU Artifact Gate 为准。

## Validation16 8GPU Artifact Gate

- val16 原始 EMA artifacts：数量/异常检查 `PASS`，但后续人工可视化发现预测结构明显塌缩，因此
  不能作为最终 artifact gate。
- val16 退出码：`0`
- val16 设备：物理 GPU 0-7，8GPU DDP。日志中注册了 `GLOBAL_RANK 0..7`。
- val16 output：
  `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/validation16_eval/`
- val16 parent log：
  `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/validation16_eval/validation16_parent.log`
- val16 metrics：
  `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/validation16_eval/metrics/version_0/metrics.csv`
- val16 artifacts：
  `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/validation16_eval/validation_artifacts/`
- artifact 数量：
  - case 目录：`16`
  - PDB 文件：`384`（每个 case `24` 个）
  - `candidate_metrics.csv`：`16`
  - `case_summary.json`：`16`
  - rank timing jsonl：`8` 个 rank 文件，共 `16` 行 timing 记录

val16 聚合值：

| metric | value |
|---|---:|
| `val/loss` | 1.2777320147 |
| `val/pdockq_loss` | 0.2133553326 |
| `val/membrane_plane_loss` | 1.8967119455 |
| `val/interaction_auxiliary_loss` | 12.1228580475 |
| `val/dockq_protein_protein_w` | 0.0681606680 |
| `val/rmsd` | 22.3182201385 |

`val/model_selection=nan`，原因是这个 16-case 小验证集上的相关性统计退化；它不是
运行失败，也没有中断 metrics/PDB 输出。

val16 没有 crop geometry、pDockQ mapping/reindex、OOM、NaN 或 sample-count mismatch
异常。`val/loss` 仍明显小于两个诊断 loss 之和，说明 validation 也保持“计算并记录，
但不进入 val total”的语义。

## Validation16 No-EMA 几何复查

- 问题：`validation16_eval/validation_artifacts` 里的 candidate PDB 预测整体团成一团。
- 根因定位：
  - 旧 validation 在 `on_validation_epoch_start()` 默认把模型切到 EMA 参数。
  - 这次 smoke 只有 50 个 train step，`ema_decay=0.999` 下 EMA 对早期权重严重滞后。
  - step-50 checkpoint 中 model 权重和 EMA 权重存在大幅差异；旧 artifacts 使用 EMA 后出现结构塌缩。
- 修复：
  - 新增 `settings.validation_use_ema_weights`，默认 `true` 保持旧语义。
  - `validation10_eval.yaml` 和 `validation16_eval.yaml` 显式设置
    `validation_use_ema_weights: false`。
  - 修复新增 `crop_segment_id` 的 padding 接入；否则真实 token 数小于 768 时会触发
    `relpos` 维度错误。
- 有效复查目录：
  `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/validation16_eval_noema_rerun1/`
- 有效 artifacts：
  `experiments/_runtime/8gpu_model_v2_fixed_crop_pdockq_auxlog_20260730_smoke80/validation16_eval_noema_rerun1/validation_artifacts/`
- 运行结果：
  - 退出码：`0`
  - 设备：物理 GPU 0-7，8GPU DDP
  - case 目录：`16`
  - PDB 文件：`400`（`384` 个 candidate + `16` 个 GT）
  - `candidate_metrics.csv`：`16`
  - `case_summary.json`：`16`
- 几何对照：
  - 旧 EMA candidate_01 的 `pred_rg / gt_rg` 均值：`0.356`，范围 `0.216-0.525`
  - no-EMA candidate_01 的 `pred_rg / gt_rg` 均值：`0.994`，范围 `0.853-1.139`
  - `1aig_1` 的 A:46->53 gap 两端 CA 距离：旧 EMA `1.559 Å`，no-EMA `44.919 Å`

结论：这次“预测团成一团”的直接原因是 smoke step-50 validation 使用了 EMA 权重；
no-EMA validation 的 PDB 尺度恢复正常。当前应以
`validation16_eval_noema_rerun1` 作为 smoke artifact gate。

## Validation 慢点定位

- 这次 val16 不是单卡 bug：DDP 注册了 8 个 rank，每个 rank 各处理 2 个样本。
- 不是 PDB 写盘瓶颈：每个 case 写 24 个 PDB 的平均时间约 `0.926s`。
- 主成本来自完整 inference rollout：`no_full_rollout_samples=24`，每个 candidate
  使用 200 步 diffusion；16 条 timing 记录中 `forward_seconds` 平均约 `106.664s/case`。
- metrics 是第二成本：`metrics_seconds` 平均约 `20.495s/case`，最大约 `29.924s/case`。
- artifact confidence 已改成轻量导出，只记录 pLDDT/pDockQ 这类标量，不再触发完整
  sample ranking；此前 full confidence ranking 在部分 crop case 上会触发
  `chain_ptm` 空维度错误，该失败尝试已归档在
  `validation16_eval/_aborted_attempts/full_confidence_artifact_failure_20260730/`。

## Reader Inventory 对齐

- 旧 Q-BioLiP manifest 比当前 reader 多 29 个 ID：train 20、val 9。
- 按用户要求没有重新 split，只移除这 29 个不可读 ID。
- Q-BioLiP active split：train `8685`、val `1307`、总计 `9992`。
- active manifest、CLI 副本、train/val ID 列表和 sampling weights 已同步。
- Q-BioLiP 和 AFDB 的 `manifest - reader`、`reader - manifest` 均为 0；
  后续训练不再打印这条 manifest warning。

## 结果

- train50 gate：`PASS`
- validation10 初筛：`PASS`
- validation16 8GPU 原始 EMA 数量/异常 gate：`PASS`，但几何复查 `FAIL`
- validation16 8GPU no-EMA artifact gate：`PASS`
- crop geometry error（train50）：`0`
- pDockQ mapping/reindex error（train50）：`0`
- crop geometry error（val10）：`0`
- pDockQ mapping/reindex error（val10）：`0`
- crop geometry error（val16）：`0`
- pDockQ mapping/reindex error（val16）：`0`
- formal gate：待使用 no-EMA validation artifacts 重新确认
