# PhysPPI dmap-aware crop 修复记录

## 当前状态

- 状态：已实现并通过 focused 验证。
- 日期：2026-07-30。
- 目标样本：`7v6c_1`。
- 原始问题：训练到 `step=59` 时，Q-BioLiP sample `7v6c_1` 失败，错误为 retained chain `B` 缺少 4 个 finite、non-collinear representative atoms，导致 mandatory monomer geometry 构造失败。
- 结论：`7v6c_1` 原始 PDB 和 monomer 数据没有缺链问题。问题来自旧 crop 路径可能把 retained protein chain 裁得过短或几何退化。修复后 `7v6c_1` 可以完整走过 dataset `ds[0]`，不再触发 chain B geometry 错误。

## 核心结论

- 不应该把 `7v6c_1` blacklist。
- 不应该修改通用 OF3 crop 原语。
- PhysPPI 需要自己的两层 crop 约束：
  - 第一层 retained-chain view 必须基于 dmap 连通性选链。
  - 第二层 token crop 必须优先保留 interface，并保证每条 retained protein chain 至少保留 `15AA`。
- retry 只应该 retry 第二层 token crop，不能每次 retry 都重新抽 retained-chain view。
- 无有效 dmap 的样本不应该进入训练 datapoint cache；index-time 直接 warning 后过滤。

## 最终策略

### 第一层 retained-chain view

- 输入：`sample.chain_sequences`、`sample.dmap`。
- 目标：从 dmap chain graph 里选一个连通 retained view。
- 选链优先级固定为 `4 -> 3 -> 2`，不再随机选 `2..4`。
- 4 链候选要求：候选链 sequence length 总和 `<=768`。
- 3 链候选要求：候选链 sequence length 总和 `<=768`。
- 2 链候选要求：必须是 dmap edge pair，但不要求总长度 `<=768`，因为后续一定还会走第二层 token crop。
- 候选生成方式：
  - 从排序后的 dmap edges 开始。
  - 通过 frontier expansion 生成 connected 3-chain candidates。
  - 再从 3-chain candidates 扩展到 connected 4-chain candidates。
  - 候选统一 canonicalize 成 sorted tuple。
  - 每个 size 的候选上限为 `1024`，超出时用 occurrence RNG 做确定性 cap。
- 如果 resource-triggered sample 没有任何 connected dmap pair，抛 `PhysPPIDataError("missing_connected_dmap_pair", ...)`，不再 fallback 到 `first 4 chains`。

### index-time dmap preflight

- 构建 `datapoint_cache` 时读取 sample dmap。
- 如果 `sample.dmap` 为空，或者没有任何有效 inter-chain dmap edge，则排除该 sample。
- 只打印一次 warning，包含排除数量和前若干 sample id。
- filtering 后再解析 sampling weights，保证权重和最终 sample id 对齐。
- runtime 阶段如果仍然遇到 retained view 没有可映射 dmap residues，则视为 index invariant 被破坏，抛硬错误。

### 第二层 PhysPPI interface-preserving token crop

- 默认开启，不修改通用 `openfold3/core/data/primitives/structure/cropping.py`。
- 新 crop 在 PhysPPI adapter/runtime 内部完成。
- token/residue mapping 使用现有 prior adapter 口径：
  - key: `(chain_id, res_id)`
  - value: 一个或多个 `token_id`
- residue priority 使用 dmap 最小 contact distance：
  - 距离越小，优先级越高。
  - priority tie 使用 deterministic jitter。
- crop priority：
  1. 每条 retained protein chain 至少保留 `15AA`。
  2. 在满足 15AA floor 后，尽量保留 dmap interface residues。
  3. 剩余 token budget 用于 interface-near context。
- 当 `min15 + all dmap <= token_budget` 时，应完整保留 dmap residues。
- 当 dmap residues 超过预算时，先保住每条 chain 的 15AA anchors，再按最小 dmap distance 保留更多 interface residues。
- 如果某条 chain 的 dmap residues 少于 15 个，用同链、距离该 chain interface 最近的 non-interface residues 补足 15AA。
- crop mask 不允许超过 `token_budget`。

### deterministic retry 和 validation guard

- retained-chain view 在一个 sample occurrence 内固定。
- retry 只作用于第二层 token crop。
- 默认 retry 次数：`16`。
- 每次 attempt seed 从 occurrence seed 和 attempt index 派生，不使用全局非确定性 `np.random` 状态。
- 每次 crop 后都显式校验：
  - 每条 retained protein chain 是否至少有 `15AA`。
  - `build_monomer_input_geometry()` 是否能构造 mandatory monomer geometry。
- 如果 16 次都失败，抛 `PhysPPIDataError("physppi_crop_retry_exhausted", ...)`，错误信息包含：
  - sample id
  - retained chains
  - attempt count
  - cropped residue counts
  - dmap residue counts
  - monomer geometry representative count / rank，若当次已经有 monomer inputs

## 代码改动

- `openfold3/projects/of3_all_atom/config/dataset_configs.py`
  - 新增 PhysPPI crop defaults：
    - `interface_preserving_crop: true`
    - `min_residues_per_retained_protein_chain: 15`
    - `geometry_crop_retry_attempts: 16`
    - `retained_view_candidate_cap: 1024`

- `openfold3/projects/of3_all_atom/physppi/data/adapters/chain_views.py`
  - 实现 dmap-connected retained-chain candidate expansion。
  - 实现 `4 -> 3 -> 2` priority selection。
  - 实现 3/4-chain length filter 和 2-chain no-length-cap 逻辑。
  - 实现 deterministic candidate cap。
  - 去掉 isolated/no-edge 的 `first 4 chains` fallback。

- `openfold3/projects/of3_all_atom/physppi/data/adapters/_runtime.py`
  - 新增 dmap preflight filtering。
  - 新增 PhysPPI interface-preserving crop mask builder。
  - 新增 crop attempt seed 派生。
  - 新增 retryable crop error 分类。
  - 新增 cropped residue floor validation。
  - 新增 crop exhaustion 诊断信息。
  - `build_sample_features()` 改成 retained view 固定、只 retry 第二层 crop。

- 测试文件：
  - `openfold3/tests/projects/of3_all_atom/physppi/data/test_chain_views.py`
  - `openfold3/tests/projects/of3_all_atom/physppi/data/test_template_crop.py`
  - `openfold3/tests/projects/of3_all_atom/physppi/data/test_qbiolip10k_dataset.py`

## 操作结果

### focused tests

命令：

```bash
LD_LIBRARY_PATH=/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/cuequivariance_ops/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cublas/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib \
PYTHONPATH=/101045/jinyu/PhysPPI_jinyu/PhysPPI:${PYTHONPATH:-} \
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python -m pytest \
  openfold3/tests/projects/of3_all_atom/physppi/data/test_chain_views.py \
  openfold3/tests/projects/of3_all_atom/physppi/data/test_template_crop.py \
  openfold3/tests/projects/of3_all_atom/physppi/data/test_qbiolip10k_dataset.py -q
```

结果：

- `51 passed, 16 warnings`

覆盖点：

- first-layer retained-chain view 优先 4/3/2。
- 3/4-chain 的 `<=768` length filter。
- 2-chain dmap pair 不受 768 限制。
- no-edge resource-triggered sample 抛错。
- candidate cap deterministic。
- dmap preflight 会过滤无效 dmap sample，并保持 sampling weights 对齐。
- interface-preserving crop 保留 dmap residues。
- dmap 超预算时先保每链 15AA，再按 dmap distance 裁剪。
- context 使用 interface-near residues。
- 同一 `(chain_id, res_id)` 多 token 时一起保留。
- 同 seed crop mask 可复现。
- retry 不重新抽 retained-chain view。
- 16 次失败后错误带诊断信息。

### AFDB regression tests

命令：

```bash
LD_LIBRARY_PATH=/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/cuequivariance_ops/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cublas/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib \
PYTHONPATH=/101045/jinyu/PhysPPI_jinyu/PhysPPI:${PYTHONPATH:-} \
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python -m pytest \
  openfold3/tests/projects/of3_all_atom/physppi/data/test_afdb10k_dataset.py \
  openfold3/tests/projects/of3_all_atom/physppi/data/test_afdb10k_reader.py -q
```

结果：

- `14 passed, 16 warnings`

### `7v6c_1` 真实 sample smoke

配置来源：

- `experiments/_runtime/8gpu_model_v2.yaml`
- 强制 `physppi.sample_ids=["7v6c_1"]`
- 走真实 `QBioLiP10kDataset(cfg)` 和完整 `ds[0]`

结果：

```json
{
  "ok": true,
  "sample_chains": ["A", "B"],
  "datapoint_cache": ["7v6c_1"],
  "cropped_chains": ["A", "B"],
  "cropped_residue_counts": {
    "A": 508,
    "B": 260
  },
  "num_atoms": 6133,
  "num_tokens": 768
}
```

结论：

- `7v6c_1` 不再触发 `retained chain 'B' lacks four finite, non-collinear representative atoms`。
- chain B 在 crop 后仍有 `260` 个 residues，不是退化成 1 个 residue。
- mandatory monomer geometry 已在完整 feature build 中通过。

### 静态检查

执行结果：

- `git diff --check`：通过，touched files 无 whitespace error。
- `py_compile`：通过，touched Python files 无语法错误。

## 验收结论

- `7v6c_1` 的 crop geometry 崩溃已针对性修复。
- 新逻辑默认开启，覆盖 Q-BioLiP/AFDB train 和 validation 的 PhysPPI adapter 路径。
- 通用 OF3 crop 行为未改。
- 对于 PhysPPI retained protein chains，新路径会尽力保证每条链至少 `15AA`；如果无法满足，会在 deterministic retry 后抛出带诊断信息的数据错误。
- dmap interface residues 在不违反 15AA floor 和 token budget 的前提下优先保留。
- DDP/restart 相关随机性由 occurrence seed 和 attempt index 派生，retained view selection 和 crop retry 可复现。

## 仍需注意

- 本次验证包含 focused tests、AFDB regression tests、真实 `7v6c_1` dataset smoke 和静态检查。
- 本次没有重新启动完整 8GPU DDP 训练。恢复训练后应观察前几百 step，确认没有新的 data-error 样本，以及吞吐没有出现不可接受的下降。
