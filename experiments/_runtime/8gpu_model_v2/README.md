# PhysPPI 8-GPU Training v2 - Length Buckets + Identity Loss Weight

> **目标目录**: `experiments/_runtime/8gpu_model_v2/`
> **基线**: `experiments/_runtime/4gpu_model_raw/` 和 `experiments/_runtime/4gpu_model_v1/`
> **GPU**: 8 ranks, `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
> **核心变化**: 不再用 identity score PT 选训练 case；改为长度分桶 + 动态 local batch，并把 identity score 作为 loss weight。

## 0. 当前结论

v2 不应该继续把所有 case 固定 pad 到 `token_budget=768` 后以 `batch_size=1` 训练。

推荐设计：

1. 每个 case 仍最多 crop 到 768 token。
2. 长度小于 768 的 case 只 pad 到所在 bucket 上界，例如 256/384/512/640/768。
3. 每个 optimizer step 只从一个长度 bucket 取样，所有 8 个 rank 使用相同的 bucket 和相同 local batch size。
4. local batch size 由实测成本模型决定；`001_256` 已实测通过 `B=8`，其他 bucket 第一版仍保守使用已测的 `1/2/4`。
5. `sampling_weights.pt` 不再作为 sampler 概率；改为 `identity_loss_weight` 乘到 per-case loss。
6. Q-BioLiP/AFDB 的 dataset loss weight 继续保留：Q-BioLiP `1.0`，AFDB `0.8`。
7. 动态 batch 需要显式记录并可选择启用 `length_loss_weight`。严格 case-uniform 目标下 `length_loss_weight=1`；如果要抑制短序列大 batch 带来的训练偏向风险，第一版使用按 bucket batch size 反比的 length weight，并把均值归一到 1。

## 1. 正式长度调研

raw/v1 的训练输入形状固定到 768。mask 只会避免 padding token 进入有效 loss，它不会回收 pair feature、attention、trunk 中间张量的显存和计算。

这次调研直接读两组预计算 PT：

- AFDB: `/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/AFDB-10k-1/chunk*/{case_id}_seq.pt`
- Q-BioLiP: `/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/Q-BioLiP-10k-new/chunk*/{case_id}_seq.pt`
- train IDs: `experiments/split_training_val/split_manifest.json`
- old identity weight mass: `/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/sampling_weights.pt`

每个 case 的长度定义为：

```text
L_total = sum_chain sum(seq_pt[chain]["mask"])
L_crop  = min(L_total, 768)
```

调研产物：

- `afdb_length_bins.png`
- `qbiolip_length_bins.png`
- `v2_bucket_batch_plan.png`
- `afdb_length_bins.csv`
- `qbiolip_length_bins.csv`
- `length_research_summary.json`
- `/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/seq_len_bin_label.pt`

### 1.1 AFDB length bins

![AFDB length bins](afdb_length_bins.png)

AFDB train 的 9,314 个 manifest train ID 全部找到 `_seq.pt`。它是 homodimer 数据，case 总长度主要集中在 641-1536，57.8% 的 case 在 768 以内。

| 指标 | 值 |
|---|---:|
| train IDs with `_seq.pt` | 9,314 / 9,314 |
| min / median / mean / max | 60 / 720 / 790.4 / 2,580 |
| P75 / P90 / P95 / P99 | 1,004 / 1,494 / 1,756.7 / 2,213.2 |
| `L_total <= 768` | 5,381 cases, 57.8% |
| unweighted fixed-768 pair waste | 31.3% |
| old identity-weighted fixed-768 pair waste | 32.9% |

AFDB 的 identity weights 总和是 14,925.1，均值 1.60。这说明旧 identity-weighted sampler 会明显提高 AFDB 的抽样质量；v2 改成 identity loss weight 后，必须把这个 weight 乘进 loss，不能丢掉。

### 1.2 Q-BioLiP length bins

![Q-BioLiP length bins](qbiolip_length_bins.png)

Q-BioLiP train manifest 有 8,705 个 ID，其中 8,685 个找到 `_seq.pt`，缺 20 个。这个缺失和 raw handoff 中提到的 reader/manifest 缺失 ID 是同一类覆盖问题，v2 sampler 构建 metadata 时必须显式报出并决定跳过还是修 manifest。

| 指标 | 值 |
|---|---:|
| train IDs with `_seq.pt` | 8,685 / 8,705 |
| missing `_seq.pt` | 20 |
| min / median / mean / max | 59 / 1,298 / 1,977.4 / 20,320 |
| P75 / P90 / P95 / P99 | 2,472 / 4,049.6 / 5,384.4 / 9,498.6 |
| `L_total <= 768` | 1,911 cases, 22.0% |
| unweighted fixed-768 pair waste | 12.8% |
| old identity-weighted fixed-768 pair waste | 28.0% |

这里和 `/101045/jinyu/PPI/Prior_out/training/Q-BioLiP-10k-new/overview.md` 的口径不同。overview 里的 `Residue Length Distribution` 是 per-chain 统计，69,778 条链分布比较均匀，median chain length 是 215 aa；这里的 `L_total` 是 per-case 把所有链的 `sum(mask)` 加起来。Q-BioLiP 平均每个 case 约 7 条链，所以 per-case median 变成 1,298，只有 22.0% 的 train case 在 768 token 以内。

Q-BioLiP 的 per-case count 分布偏长，但 old identity weight mass 明显向短/中等长度移动。v2 不再用 identity weight 采样后，短 case 不会天然更常出现，所以长度 bucket 的 case-count 采样和 identity loss weight 必须同时记录，便于比较“吞吐提升”和“训练目标是否变化”。

### 1.3 合并结论

按 `_seq.pt` 覆盖到的 17,999 个 train case：

| 指标 | 值 |
|---|---:|
| manifest train IDs | 18,019 |
| train IDs with `_seq.pt` | 17,999 |
| missing `_seq.pt` | 20 |
| `L_total <= 768` | 7,292 cases, 40.5% |
| unweighted fixed-768 pair waste | 22.3% |
| old identity-weighted fixed-768 pair waste | 32.1% |
| combined identity weight mean | 1.001 |

这里的 22.3% 是 v2 uniform/count-based sampler 下更接近的 pair 浪费估计；32.1% 是 old identity-weighted sampler 下的估计。由于 v2 把 identity score 从 sampling 挪到 loss，实际训练吞吐预期应以 22.3% 作为更保守的起点。

## 2. 成本模型与 `L^2` 简化

**当前状态**：已经在当前 PhysPPI model 上跑过 8GPU 真实 step benchmark，并补齐了 full-rank `B,L` 测量矩阵。当前成功点覆盖 `N_pad=256/384/512/640/768` 的 `B=1`，`256/384` 的 `B=2/4`，以及 `512` 的 `B=2`。因此 README 末尾已经给出实测拟合的 `t0/a0/a1/a2` 和第一版 batch 计划。

用户提出的近似是对的：

```text
C_sample(L) = a0 + a1 * L + a2 * L^2
```

如果一个 local batch 有 `B` 个相近长度的 case：

```text
T(B, L) = t_step_fixed + B * C_sample(L)
```

为了让不同长度 bucket 的 step 时间接近：

```text
B(L) = floor((T_target - t_step_fixed) / (a0 + a1 * L + a2 * L^2))
```

所以 `B` 和 `a0 + a1 L + a2 L^2` 成反比。

如果 profiling 证明 `L^2` 项占主导，即：

```text
a2 * L^2 >> a0 + a1 * L
```

就可以简化为：

```text
B(L) ~= floor(C / (a2 * L^2))
```

其中 `C = T_target - t_step_fixed` 是每 rank 每 step 愿意给这个 batch 的目标计算预算。若把 `C=1`，并设置 `a2=0.0001`：

| L | `a2 * L^2` | `B = 1 / (a2 * L^2)` |
|---:|---:|---:|
| 10 | 0.01 | 100 |
| 20 | 0.04 | 25 |
| 50 | 0.25 | 4 |
| 100 | 1.00 | 1 |
| 256 | 6.5536 | 0.153 |

这正是用户说的例子：`L=10` 时 `B=100`。但训练里 `B` 至少要是 1，因此直接用 `C=1` 不适合 768-token 参考点。更实用的写法是选参考长度 `L_ref=768` 且希望 `B_ref=1`：

```text
C = a2 * L_ref^2
B(L) ~= floor(B_ref * (L_ref / L)^2)
```

此时 `a2` 会被参考预算吸收，batch 计划只取决于相对长度：

| `N_pad` | `(768 / N_pad)^2` | 无上限 local batch | v2 初始 local batch |
|---:|---:|---:|---:|
| 256 | 9.00 | 9 | 4 |
| 384 | 4.00 | 4 | 4 |
| 512 | 2.25 | 2 | 2 |
| 640 | 1.44 | 1 | 1 |
| 768 | 1.00 | 1 | 1 |

第一版不建议全局无差别放开大 batch。`001_256` 已经实测 `B=8` 可跑；其他 bucket 仍建议先限制在已测范围内，因为真实显存还受 atom count、MSA depth、template、48 diffusion candidates、DockQ/permutation alignment 和 loss 中间张量影响。

正式计划要先跑 benchmark，再拟合完整模型：

```text
T(B, L) ~= t0 + B * (a0 + a1 * L + a2 * L^2)
M(B, L) ~= m0 + B * (m1 * L + m2 * L^2 + m_atom * atom_count)
```

最少需要测这些点：

| `N_pad` | local batch candidates |
|---:|---|
| 256 | 1, 2, 4, 8 |
| 384 | 1, 2, 4 |
| 512 | 1, 2, 4 |
| 640 | 1, 2 |
| 768 | 1 |

每个点至少跑 warmup 2 step + measured 5 step，记录：

- DataLoader wait time
- forward time
- loss/DockQ/permutation/RASA time
- backward time
- optimizer + DDP time
- CUDA peak allocated/reserved memory
- `N_pad`, atom count, MSA depth, template count, diffusion candidates

只有当 `a2 * L^2` 在拟合中确实主导，并且 memory model 也允许，才可以把 batch 规则简化为 `B(L) ~= C / (a2 L^2)`。

## 3. 实测分组计划

下面是 benchmark 后的第一版训练分组计划。v2 bucket 使用 crop 后的真实 `N_crop = min(L_total, 768)`，并向 64 对齐。这里的 `641_768` 不是说原始 case 长度大多刚好在 768 附近，而是因为所有 `L_total > 640` 的 case 在训练 crop 视角都会进入这个 bucket；其中包括全部 `L_total > 768` 的长 case。

![v2 bucket batch plan](v2_bucket_batch_plan.png)

| bucket | `N_pad` | 覆盖 case 数（按 `N_crop`） | benchmark batch candidates | 第一版 local B | 来源 |
|---|---:|---:|---:|---:|---|
| `001_256` | 256 | 1,050 | 1, 2, 4, 8 | 8 | `B=8` 已实测通过 |
| `257_384` | 384 | 1,308 | 1, 2, 4 | 4 | `B=4` 已实测通过，预测 step 低于 768/B1 |
| `385_512` | 512 | 1,681 | 1, 2, 4 | 2 | `B=2` 已实测通过；`B=4` 暂不开放 |
| `513_640` | 640 | 1,422 | 1, 2 | 1 | 目前只测 `B=1`，先保守不开放 2 |
| `641_768` | 768 | 12,538 | 1 | 1 | 参考 bucket |

这个表以 README 末尾的 full-rank fit 和成功测量点为准。`a0/a1`、atom count、optional prior、loss/metric 开销都不是零，所以第一版不要按纯 `L^2` 公式继续外推到更大 batch。

不要把 200-token case 和 768-token case 放在同一个 batch。collator 会 pad 到 batch 最大长度，混放会把短 case 的收益吃掉。

当前最重要的结论是：`a0/a1/a2` 已经不是凭公式假设出来的，而是用真实 8GPU model step 拟合出来的。按“只采用已实测 bucket batch”的原则，第一版动态 batch 可用 `B=8/4/2/1/1`；`benchmark_runner.py summarize --max-batch-size 8` 的纯模型推荐会给 `384 -> B=5`，但 384 的 B5 没有实测，第一版不要采用。必须同时把 sampler trace、per-case loss、identity weight、length weight 都落盘，否则后面无法判断吞吐提升是否改变了训练目标。

补充：`001_256_B8` 后续实测已经通过，median step 35.62s，peak reserved 65.76GB，低于 `641_768_B1` 的 71.17GB；但该点 wall time 约 555s，首 batch 数据/特征准备明显变重。当前已经生成的 `seq_len_bin_label.pt` 仍是旧的 256 bucket `local_batch_size=4` 标签；如果正式采用 B8，需要重生成该文件里的 `local_batch_size` 和 `length_loss_weight_inverse_batch`。

## 4. Sampling 语义

v1/raw 当前语义：

```text
P(case i) = identity_weight_i / sum(identity_weight)
loss_i 不额外乘 identity_weight_i
```

v2 改为：

```text
P(case i) = 1 / N_available       # 或先按 length bucket 的 case count 选 bucket
loss_i *= identity_loss_weight_i  # sampling_weights.pt 中的 mean-normalized 权重
```

如果 `sampling_weights.pt` 的权重均值为 1，则：

```text
E_uniform[identity_weight_i * loss_i]
= (1 / N_available) * sum_i identity_weight_i * loss_i
= sum_i (identity_weight_i / sum_j identity_weight_j) * loss_i
```

如果可用训练池里的 identity weight 均值为 1，这和原来的 identity-weighted sampling 在数学期望上一致。本次 `_seq.pt` 覆盖到的 17,999 个 train case 中，combined identity weight mean 为 1.001，足够接近；缺失的 20 个 Q-BioLiP ID 需要先处理，否则不要把 18,019 直接当作 sampler 分母。

注意：不能既按 identity weight 抽样，又把同一个 weight 乘到 loss；那会把权重平方化。

## 5. Bucket Sampler 设计

已经生成训练侧长度分组标签：

```text
/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/seq_len_bin_label.pt
```

这个 PT 不只是单个 label tensor，而是一个 schema 化 dict。训练时 sampler 应读取：

- `case_to_label[case_id]["bucket"]`
- `case_to_label[case_id]["n_pad"]`
- `case_to_label[case_id]["local_batch_size"]`
- `case_to_label[case_id]["identity_loss_weight"]`
- `case_to_label[case_id]["length_loss_weight_inverse_batch"]`
- `bucket_to_case_ids[bucket]`

当前文件摘要：

| 指标 | 值 |
|---|---:|
| schema | `physppi_seq_len_bin_labels_v1` |
| covered train cases | 17,999 |
| expected train IDs | 18,019 |
| missing `_seq.pt` | 20 |
| AFDB covered | 9,314 |
| Q-BioLiP covered | 8,685 |
| mean identity loss weight | 1.001 |
| mean inverse-batch length weight | 1.000 |

| bucket | `N_pad` | cases | local B | inverse-batch `length_loss_weight` |
|---|---:|---:|---:|---:|
| `001_256` | 256 | 1,050 | 4 | 0.292 |
| `257_384` | 384 | 1,308 | 4 | 0.292 |
| `385_512` | 512 | 1,681 | 2 | 0.585 |
| `513_640` | 640 | 1,422 | 1 | 1.170 |
| `641_768` | 768 | 12,538 | 1 | 1.170 |

推荐第一版 sampler：

1. 离线或 DataModule setup 时为每个 train case 建立 metadata：
   - dataset key: `qbiolip` / `afdb`
   - case id
   - model token count
   - bucket id
   - identity loss weight
   - dataset loss weight
   - length loss weight
2. 选择 bucket 时按 bucket 的 case count 权重选：

   ```text
   P(bucket b) = num_cases_in_bucket_b / N_available
   ```

3. bucket 内 uniform with replacement 抽样。
4. batch 内 per-case loss 先保留 batch 轴，乘：

   ```text
   total_case_weight = identity_loss_weight * dataset_loss_weight * length_loss_weight
   ```

5. 对 batch 求 `mean(case_weight * loss)`，再给 DDP all-reduce。

启用任意 per-case weight 后，每个 optimizer step 的期望梯度是：

```text
E[g_step] = (1 / N_available) * sum_i identity_weight_i * dataset_weight_i * length_weight_i * g_i
```

当 `length_loss_weight=1` 且 bucket 按 case count 采样时，上式退化为 v1/raw 的 identity-weighted case objective。短 bucket 的 batch size 更大，主要收益是降低该 bucket 条件梯度的方差，并提高吞吐；它不会在数学期望上提高短 case 的 case-uniform 权重。若启用下面的 length-aware 方案，则是在 case-uniform 目标之外主动给长序列更高权重，用来抵消动态 batch 的方差和单位时间 exposure 差异。

## 6. Dataset / Identity / Length Weight

v2 同时存在两类权重，不能混淆：

| 权重 | 来源 | 用途 |
|---|---|---|
| `dataset_loss_weight` | YAML: Q-BioLiP `1.0`, AFDB `0.8` | 数据集级 loss 系数 |
| `identity_loss_weight` | `sampling_weights.pt` | 代替 identity sampling 的 per-case loss 系数 |
| `length_loss_weight` | bucket metadata / fitted cost model | 可选：降低短序列、提高长序列的 per-case loss 系数 |

若目标是严格复刻旧 identity-weighted sampling 的数学期望，推荐最终 per-case loss：

```text
weighted_loss_i = loss_i * dataset_loss_weight_i * identity_loss_weight_i
batch_loss = mean(weighted_loss_i)
```

如果后续发现 identity weight 极端值导致单个 batch loss 尺度抖动，可以考虑 weighted mean：

```text
batch_loss = sum(weighted_loss_i) / sum(dataset_loss_weight_i * identity_loss_weight_i)
```

两者方向接近但尺度不同。第一版建议先用 `mean(weighted_loss_i)` 保持期望语义一致，并记录 `case_weight_sum`、`case_weight_mean`、`case_weight_max`；是否切换 weighted mean 需要单独作为实验变量。

## 7. Length-aware Loss Weight

这里要分清两个目标。

**目标 A：严格 case-uniform / old identity sampling 等价。**

如果 sampler 每个 optimizer step 按 case count 选择 bucket：

```text
P(bucket b) = n_b / N_available
```

并且 batch loss 用 batch mean：

```text
batch_loss = mean_i(dataset_weight_i * identity_weight_i * loss_i)
```

那么动态 local batch 不会改变 per-step 期望目标。此时数学上最干净的 length weight 是：

```text
length_loss_weight_i = 1
```

这和“所有 case 都 pad 到 768、local B=1”的 case-level 目标等价；区别只在短 bucket 一步看更多 case，梯度方差更低，吞吐更高。

**目标 B：抑制短序列大 batch 的优势，主动补偿长序列。**

如果担心短序列因为 `B` 更大而在有限训练时间内学得更稳，建议把 length weight 作为显式 ablation，而不是混在 identity weight 里。第一版采用和当前 batch 计划绑定的保守公式：

```text
w_len_raw(bucket b) = B_ref / B_b
B_ref = 1
length_loss_weight_b = w_len_raw(bucket b) / mean_train(w_len_raw)
```

当前 bucket 计划下：

| bucket | `N_pad` | local B | `w_len_raw = 1 / B` | mean-normalized `length_loss_weight` |
|---|---:|---:|---:|---:|
| `001_256` | 256 | 4 | 0.25 | 0.292 |
| `257_384` | 384 | 4 | 0.25 | 0.292 |
| `385_512` | 512 | 2 | 0.50 | 0.585 |
| `513_640` | 640 | 1 | 1.00 | 1.170 |
| `641_768` | 768 | 1 | 1.00 | 1.170 |

这个设计的直觉是：一个短 bucket step 里每个 rank 一次处理 4 个 case，则每个短 case 的额外 length 系数先降到 `1/4`；长 bucket 仍是 `1`。再用训练集 bucket count 把均值归一到 1，避免整体 loss 尺度大幅改变。

也可以使用拟合成本模型定义更物理的权重：

```text
C_sample(L) = a0 + a1 * L + a2 * L^2
w_len_raw(L) = clip(C_sample(L) / C_sample(768), w_min, 1.0)
```

当前实测系数下，未裁剪的 cost ratio 约为：

| `N_pad` | `C_sample(N_pad) / C_sample(768)` | pure `L^2` ratio `(N_pad/768)^2` |
|---:|---:|---:|
| 256 | 0.057 | 0.111 |
| 384 | 0.174 | 0.250 |
| 512 | 0.370 | 0.444 |
| 640 | 0.646 | 0.694 |
| 768 | 1.000 | 1.000 |

这个版本更贴近 profiler，但 256/384 的权重会很小，第一版不建议直接使用未裁剪 cost ratio。若要用 cost ratio，建议设置 `w_min=0.25`，再按训练集均值归一。

生产训练建议保留三个明确配置：

```yaml
length_loss_weight:
  mode: inverse_batch_size  # choices: none, inverse_batch_size, fitted_cost_ratio, l2_ratio
  normalize_mean_to_one: true
  min_weight: 0.25
  reference_n_pad: 768
```

第一轮 v2 主实验推荐：

```text
total_case_weight_i =
  dataset_loss_weight_i
  * identity_loss_weight_i
  * length_loss_weight_bucket_i

batch_loss = mean(total_case_weight_i * per_case_loss_i)
```

必须在日志里分别写出 `dataset_loss_weight`、`identity_loss_weight`、`length_loss_weight` 和 `total_case_weight`。否则如果长序列效果提升或短序列效果下降，后面无法区分是动态 batch、identity weight，还是 length weight 导致的。

## 8. 必须先修的 batch>1 假设

当前训练代码仍有 batch size 1 假设，v2 不能只改 YAML：

1. `apply_physppi_loss_weight()` 目前只接受 rank-local 单个 scalar，需要支持 `[B]`。
2. route-c train metrics JSONL 现在按单 case 写，需要支持每个 rank 每个 step 写 `B` 行。
3. loss module 需要能返回 per-example loss，至少在 PhysPPI 的 dataset/identity weighting 之前不要先 reduce 成一个 scalar。
4. DockQ、pDockQ target、permutation alignment、RASA、membrane loss 都要跑 `B>1` 回归。
5. DDP 所有 rank 必须使用同一 local batch size，否则 backward/all-reduce 会 hang。
6. `epoch_len`、`max_steps`、scheduler、checkpoint 和 validation interval 要明确按 optimizer step 还是 processed cases 计数。

## 9. 8GPU 启动约定

v2 启动脚本应使用：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

YAML 中对应：

```yaml
data_module_args:
  batch_size: 1  # 初始 smoke 仍用 1；bucket batch sampler 接管后再改语义
  epoch_len: 4096
  global_datapoint_sampling: false
  distinct_samples_per_rank_group: false

pl_trainer_args:
  devices: 8
  num_nodes: 1
  precision: bf16-mixed
  accumulate_grad_batches: 1
```

`global_datapoint_sampling` 在 v2 应关闭或被新 sampler 替代，因为 identity score 不再用于抽样。

## 10. Benchmark runner

新增 runner：

```text
experiments/_runtime/8gpu_model_v2/benchmark_runner.py
```

它负责四件事：

1. 生成候选矩阵：`N_pad=256/384/512/640/768` 与 local batch candidates。
2. 从真实 `_seq.pt` 和 manifest 中为每个 bucket 选择固定 case，写入 `benchmark_cases.json`。
3. 为单个 candidate 生成固定 `sample_ids` 的 YAML，并调用真实 `run_openfold train`。
4. 从 `benchmark_results_step_model.jsonl` 拟合成本曲线，生成 `benchmark_fit_summary_step_model.json`，并可把拟合结果追加回 README。

### 10.1 准备固定 case

```bash
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python \
  experiments/_runtime/8gpu_model_v2/benchmark_runner.py prepare-cases \
  --output experiments/_runtime/8gpu_model_v2/benchmark_cases.json \
  --cases-per-dataset-bucket 4
```

当前已生成 `benchmark_cases.json`。每个 bucket 会分别选 AFDB 和 Q-BioLiP 的固定 case，避免 benchmark 过程中 case pool 随机漂移。

### 10.2 查看候选矩阵

```bash
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python \
  experiments/_runtime/8gpu_model_v2/benchmark_runner.py matrix
```

候选点：

| `N_pad` | local batch candidates |
|---:|---|
| 256 | 1, 2, 4, 8 |
| 384 | 1, 2, 4 |
| 512 | 1, 2, 4 |
| 640 | 1, 2 |
| 768 | 1 |

### 10.3 生成训练长度 bin 标签

```bash
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python \
  experiments/_runtime/8gpu_model_v2/benchmark_runner.py write-bin-labels \
  --output /101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/seq_len_bin_label.pt
```

这个命令会重新扫描 train manifest 里的 AFDB/Q-BioLiP `_seq.pt`，写出 `case_to_label` 和 `bucket_to_case_ids`。训练启动时不应该再现场扫描所有 `_seq.pt`，直接读这个文件即可。

### 10.4 运行一个真实 candidate

```bash
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python \
  experiments/_runtime/8gpu_model_v2/benchmark_runner.py run-candidate \
  --bucket 001_256 \
  --n-pad 256 \
  --batch-size 1 \
  --warmup-steps 2 \
  --measured-steps 5 \
  --timeout-seconds 1800 \
  --results-jsonl experiments/_runtime/8gpu_model_v2/benchmark_results_step_model.jsonl
```

runner 会写：

- candidate YAML: `benchmark_runs/{bucket}_B{B}/{bucket}_B{B}.yaml`
- train log: `benchmark_runs/{bucket}_B{B}/benchmark.log`
- result row: `benchmark_results_step_model.jsonl`

状态说明：

| status | 含义 |
|---|---|
| `ok` | candidate 运行完成，可用于拟合 |
| `failed` | `run_openfold` 非零退出，保留 log 路径 |
| `timeout` | 超过 `--timeout-seconds`，runner 会终止 DDP 进程组并记录该点 |

当前正式实测结果保存在 `benchmark_results_step_model.jsonl`。已有 10 个成功点覆盖 `N_pad=256/384/512/640/768` 的 `B=1`，`256/384` 的 `B=2/4`，以及 `512` 的 `B=2`。为让 benchmark 能测到真实 B>1 step，`benchmark_worker.py` 只在 benchmark 进程内绕过/修正了若干 batch-size-1 假设；这些修正不等价于生产训练已完整支持 B>1。

### 10.5 拟合系数并写回 README

收集足够多 `status=ok` 的真实结果后运行：

```bash
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python \
  experiments/_runtime/8gpu_model_v2/benchmark_runner.py summarize \
  --measurements experiments/_runtime/8gpu_model_v2/benchmark_results_step_model.jsonl \
  --output experiments/_runtime/8gpu_model_v2/benchmark_fit_summary_step_model.json \
  --readme experiments/_runtime/8gpu_model_v2/README.md
```

该命令会拟合：

```text
T(B, L) ~= t0 + B * (a0 + a1 * L + a2 * L^2)
M(B, L) ~= m0 + B * (m1 * L + m2 * L^2 + m_atom * atom_count)
```

并在 README 末尾生成 `Benchmark 拟合结果` section。当前 `time_fit` 是 full-rank：`design rank=4/4`，可用于第一版动态 batch 分组计划。`001_256_B8` 已经是成功实测点；其他 bucket 仍不要采用未测的大 batch，例如 384 的模型推荐 `B=5` 目前只能视为预测值。

## 11. 推荐实施顺序

### Phase A - 只加观测，不改训练语义

- 记录每个 occurrence 的 `model_token_count`、`N_pad`、bucket、atom count、MSA depth、identity weight。
- 记录 DataLoader wait、forward、loss、backward、optimizer、CUDA peak memory。
- 8GPU `batch_size=1` smoke，确认 v1 pDockQ 修复、checkpoint、all-rank logging 可用。

### Phase B - 动态 N，仍 batch size 1

- 把短 case 的 shape 从固定 768 改成 `N_pad = ceil_to_multiple(actual_N, 64)`。
- 保持 sampler、loss、optimizer step 语义不变。
- 对固定 case 集合做 old-768 vs dynamic-N A/B benchmark。

### Phase C - identity loss weight，仍 batch size 1

- sampler 改 uniform/count-based，不读 identity weight 做抽样。
- `sampling_weights.pt` 只进入 batch metadata。
- loss 乘 `identity_loss_weight`，验证长时间期望频率和 loss 权重统计。

### Phase D - length bucket + local batch 1/2/4/8

- 实现 deterministic distributed bucket batch sampler。
- 所有 8 rank 每个 step 使用相同 bucket 和 local batch size。
- batch 内只 pad 到 bucket 上界。
- 第一版采用已实测的 `B=8/4/2/1/1`；不要采用未实测的 384/B5 或 512/B4。
- 同时保留 `length_loss_weight=none` 和 `inverse_batch_size` 两个配置；主实验可以先跑 `inverse_batch_size`，但必须能回退到 `none` 做目标等价对照。

## 12. 验证清单

- 8 rank 都启动在 GPU `0..7`。
- 每个 rank 的 sampler trace 都包含 bucket id、local batch size、case ids、identity weights。
- `sampling_weights.pt` 不再影响 case 抽样频率；只影响 loss。
- 长时间抽样频率近似 uniform/count-based。
- `mean(identity_loss_weight)` 在训练池约为 1。
- `mean(length_loss_weight)` 在训练池约为 1，并且日志中能按 bucket 还原。
- 至少保留 `length_loss_weight=none` 和 `inverse_batch_size` 两个可复现实验配置，不能只跑加权版本。
- `B=1` 时 v2 dynamic-N 与 v1/raw 的有效 token、AtomArray、RASA、DockQ 对齐。
- `B>1` 时 loss JSONL 每个 case 一行，而不是只写 batch 平均。
- validation 仍完整遍历 Q-BioLiP 1,316 和 AFDB 686 个 val case；validation 不使用 identity loss weight。
- checkpoint 每 50 step 可保存并恢复。

## 13. 明确不做

- 不停止或覆盖 `4gpu_model_raw` / `4gpu_model_v1`。
- 不把 `sampling_weights.pt` 同时用于 sampling 和 loss。
- 不把 AFDB `0.8` 当作采样比例。
- 不在没有 profiler 的情况下直接把 `N<=256` 的 local batch 开到 8 或 16。
- 不把 validation 改成 weighted 或 sampled validation。
- 不把 `length_loss_weight` 和 bucket sampling 概率同时改成偏长；第一版只允许改 loss weight，bucket 选择仍按 case count。

## 14. 参考文件

- `experiments/_runtime/4gpu_model_raw/HANDOFF20260729.md`
- `experiments/_runtime/4gpu_model_v1/readme.md`
- `experiments/_runtime/4gpu_train_v1.yaml`
- `experiments/split_training_val/split_manifest.json`
- `experiments/_runtime/8gpu_model_v2/benchmark_runner.py`
- `experiments/_runtime/8gpu_model_v2/benchmark_worker.py`
- `experiments/_runtime/8gpu_model_v2/benchmark_cases.json`
- `experiments/_runtime/8gpu_model_v2/benchmark_results_step_model.jsonl`
- `experiments/_runtime/8gpu_model_v2/benchmark_fit_summary_step_model.json`
- `experiments/_runtime/8gpu_model_v2/afdb_length_bins.png`
- `experiments/_runtime/8gpu_model_v2/qbiolip_length_bins.png`
- `experiments/_runtime/8gpu_model_v2/v2_bucket_batch_plan.png`
- `experiments/_runtime/8gpu_model_v2/afdb_length_bins.csv`
- `experiments/_runtime/8gpu_model_v2/qbiolip_length_bins.csv`
- `experiments/_runtime/8gpu_model_v2/length_research_summary.json`
- `/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/sampling_weights.pt`
- `/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/seq_len_bin_label.pt`
- `/101045/jinyu/PPI/Prior_out/training/AFDB-10k-1/overview.md`
- `/101045/jinyu/PPI/Prior_out/training/Q-BioLiP-10k-new/overview.md`

## 15. Benchmark 拟合结果

由 `benchmark_runner.py summarize` 生成。

Full model status: full-rank fit available.

```text
t0 = 7.37732
a0 = 3.67208
a1 = -0.0390054
a2 = 0.000150307
time R2 = 0.957299
design rank = 4/4

B=1 identifiable length fit:
b0 = 2.97413
b1 = -0.0154558
b2 = 0.000135817
B=1 R2 = 0.978102
```

| candidate | `N_pad` | B | median sec | peak reserved GB |
|---|---:|---:|---:|---:|
| 001_256_B1 | 256 | 1 | 6.727 | 11.803 |
| 257_384_B1 | 384 | 1 | 21.250 | 18.168 |
| 385_512_B1 | 512 | 1 | 25.261 | 30.123 |
| 513_640_B1 | 640 | 1 | 51.733 | 45.678 |
| 641_768_B1 | 768 | 1 | 70.602 | 71.174 |
| 001_256_B2 | 256 | 2 | 16.117 | 19.650 |
| 257_384_B2 | 384 | 2 | 35.576 | 32.160 |
| 001_256_B4 | 256 | 4 | 20.192 | 35.840 |
| 257_384_B4 | 384 | 4 | 51.583 | 60.732 |
| 385_512_B2 | 512 | 2 | 47.508 | 55.074 |
| 001_256_B8 | 256 | 8 | 35.619 | 65.758 |

| `N_pad` | recommended B | time-limited B | memory-limited B | predicted sec | predicted GB |
|---:|---:|---:|---:|---:|---:|
| 256 | 8 | 8 | 8 | 35.675 | 65.823 |
| 384 | 5 | 5 | 8 | 61.666 | 77.636 |
| 512 | 2 | 2 | 4 | 53.584 | 56.866 |
| 640 | 1 | 1 | 2 | 47.652 | 47.186 |
| 768 | 1 | 1 | 1 | 69.748 | 66.276 |

## 16. 最终 v2 训练测速

2026-07-29 最终决策：

- v2 暂不启用 length bucket / dynamic batch。
- `seq_len_bin_label.pt` 保留在 `/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/seq_len_bin_label.pt`，但训练配置不读取、不使用它。
- identity score 继续只通过 `/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/sampling_weights.pt` 做 sampling，不改成 loss weight。
- diffusion `no_samples` 从默认配置改为 `24`。
- 训练使用 8GPU `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，每卡 `batch_size=1`，全局 batch 为 8。

正式配置文件：

- `experiments/_runtime/8gpu_model_v2.yaml`

为使 8GPU 训练可启动，配置里做了两个工程修正：

- `val_check_interval=512`。原因是 `epoch_len=4096` 且 8GPU 下每个 epoch 只有 512 个 train batches，`val_check_interval=1000` 会被 Lightning 直接拒绝。
- `log_every_n_steps=1`。这是为了测速时每个 step 都能写 CSV；不改变训练语义。

训练启动时发现两个高频 warning 会把 rank 输出管道打满，导致进程停在 `pipe_write`、GPU util 为 0。已将以下两处从 `logger.warning` 降为 `logger.debug`，只影响日志级别：

- `openfold3/core/data/pipelines/sample_processing/conformer.py`
- `openfold3/core/data/primitives/structure/cleanup.py`

实测 run：

- parent log: `experiments/_runtime/8gpu_model_v2/train_n24_8gpu_v3_parent.log`
- torchrun rank logs: `experiments/_runtime/8gpu_model_v2/torchrun_logs_n24_v3/`
- metrics: `experiments/_runtime/8gpu_model_v2/metrics/version_4/metrics.csv`

`metrics.csv` 成功写到 step 7。按连续 CSV 写入时间估算：

| range | wall time | step speed |
|---|---:|---:|
| step 2 -> 4 | 203.8 s / 2 steps | 101.9 s/step |
| step 4 -> 6 | 215.8 s / 2 steps | 107.9 s/step |
| step 6 -> 7 | 99.7 s / 1 step | 99.7 s/step |
| step 2 -> 7 | 519.4 s / 5 steps | 103.9 s/step |

结论：当前 v2 fixed-768、8GPU、B=1、`no_samples=24` 的 steady-state step speed 约为 `104-105 s/step`，即约 `0.0096 step/s`。按全局 batch 8 计，吞吐约 `0.077 samples/s`。启动和首批数据准备耗时明显更长，不应计入 steady-state step speed。
