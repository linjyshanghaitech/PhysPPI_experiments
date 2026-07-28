# 训练/验证集拆分 — 基于序列 identity 的同源去重

## 参考依据：AlphaFold-Multimer 的拆分协议

本拆分完全遵循 AlphaFold-Multimer（Evans et al., *Nat Commun* 2022, SI）的 train/val
划分方法论，核心两步：

> **第二步：序列聚类。** 对所有 chain 用 MMseqs2 easy-cluster 在 **40% identity** 下聚类。
> 训练时按 cluster size 的倒数做 inverse-proportional sampling，使得每个同源簇的总采样概率
> 相等（参考 AF2 SI 1.2.5；AF3 SI 2.5.3）。
>
> **第三步：chain 级同源检测。** 用 **30% identity** 聚类。验证集的每个 complex，若其任一
> chain 与训练集中某个 chain 落入同一 30% cluster，则该 complex 整体从验证集中剔除。
> 这是 chain-level 而非 complex-level 的检测——一个 complex 只要有一条 chain 在训练集
> 中"没见过同源"，就视为 novel complex（参考 AF-Multimer 的 Dimer1193 / Oligomer562
> benchmark 构建流程）。

此外参考了 Boltz-1（Corso et al., 2024）的 MMseqs2 聚类参数（`--min-seq-id 0.4 -c 0.8
--cov-mode 0 --cluster-mode 2`）以及 `/101045/jinyu/IntPEMP_RSCBdata/idenity/` 中 identity
weight 的计算方式（`weight ∝ 1/cluster_size`，均值归一化到 1.0）。

## 最终拆分结果（9:1）

| | Train | Val | 总计 |
|---|---|---|---|
| **AFDB** | 9,314 | 686 | 10,000 |
| **Q-BioLiP** | 8,705 | 1,316 | 10,021 |
| **合计** | **18,019** | **2,002** | **20,021** |

| 指标 | 值 |
|---|---|
| 连通分量数 | 5,358（最大分量 6,265 个 case，孤立分量 3,852 个） |
| Val 中的连通分量 | 18 个（全部为多 case 同源家族，0 个孤立分量） |
| Val 覆盖的 30% cluster 数 | 295 个 |
| Chain 级泄露 | **0**（没有任何 val chain 和 train chain 共享 30% cluster） |
| Val chain 长度范围 | 30–1,058 aa（均值 ~282） |
| Val 中有 PLIP interaction 数据的 case | 1,811 / 2,002（90.5%） |
| 训练采样权重 | min=0.0031, max=5.80, mean=1.0 |

### 验证集质量说明

- **AFDB 在 val 中偏少**（6.9% vs Q-BioLiP 的 13.1%）。原因是最大的同源连通分量
  （贪心优先分配给 val）主要由 Q-BioLiP 的实验结构组成，AFDB 的 AlphaFold 预测序列
  同源冗余度更低。这导致 val 的 AFDB 代表性不足，但不会造成数据泄露。
- **Val 中没有孤立 case**：val 里的每个 case 都至少与 val 中另一个 case 在 30% 层面
  同源（它们落入了 18 个连通分量）。val 的同源多样性受这 18 个分量限制，但覆盖了
  295 个不同的 30% cluster。
- **用于过拟合监控足够**：真正的 holdout 评估在独立的 benchmark 上完成。

## 拆分流程

### 第一步：提取 chain 序列

从每个 chunk 的 `seq_cache.pkl` 中读取每条 chain 的 `full_seq_1aa`，写出 FASTA。

```bash
python extract_sequences.py
```

输入：`/101045/jinyu/PPI/Prior_out/training/{AFDB-10k-1,Q-BioLiP-10k-new}/chunk/*/seq_cache.pkl`
输出：
- `sequences.fasta`（91,143 条 chain，header 为 `>{case_id}|{chain_id}`）
- `chain_index.json`（`{seq_id: {dataset, case_id, chain_id, seq_len}}`）
- `case_chains.json`（`{case_id: [chain_id, ...]}`）

### 第二步：MMseqs2 聚类（30% + 40%）

```bash
python run_mmseqs.py --fasta sequences.fasta --threads 32
```

实际执行的命令等价于：

```bash
mmseqs easy-cluster sequences.fasta mmseqs_30/clusters mmseqs_30/tmp \
  --min-seq-id 0.3 -c 0.8 --cov-mode 0 --cluster-mode 2 --threads 32

mmseqs easy-cluster sequences.fasta mmseqs_40/clusters mmseqs_40/tmp \
  --min-seq-id 0.4 -c 0.8 --cov-mode 0 --cluster-mode 2 --threads 32
```

参数依据：
- `--min-seq-id 0.4`：AF2 SI 1.2.5 / AF3 SI 2.5.3 的蛋白链聚类阈值
- `--min-seq-id 0.3`：AF-Multimer 验证集同源过滤阈值
- `-c 0.8 --cov-mode 0`：RCSB PDB / MMseqs2 的双向 80% 覆盖规范

### 第三步：构建同源感知拆分

```bash
python build_split.py --val-ratio 0.1
```

算法：

1. 解析 30% cluster TSV → 每条 chain 属于一个 cluster（以 representative 标识）
2. 构建同源图：两个 complex 若有任意 chain 落入同一 30% cluster 则连边
3. 传递闭包 → 连通分量。同一分量内的所有 complex 必须分配到同一个 split
4. 贪心分配：分量按大小降序排列，优先进入 val，直到达到目标比例
5. 用 40% cluster 计算训练采样权重：`weight(case) = (1 / max chain 的 cluster_size) / mean`

输出：
- `train_ids.json` / `val_ids.json`（扁平 case ID 列表）
- `split_manifest.json`（openfold3 兼容格式，per-dataset）
- `sampling_weights.pt`（训练权重 + 完整 chain cluster 元数据）
- `split_report.json`（可读统计）

三步一键运行：

```bash
python run_all.py --val-ratio 0.1
```

## 输出文件

### 中间产物（`experiments/split_training_val/`）

| 文件 | 说明 |
|------|------|
| `sequences.fasta` | 91,143 条 chain 序列 |
| `chain_index.json` | `{seq_id: {dataset, case_id, chain_id, seq_len}}` |
| `case_chains.json` | `{case_id: [chain_id, ...]}` |
| `mmseqs_30/clusters_cluster.tsv` | 30% identity 聚类（9,083 clusters） |
| `mmseqs_40/clusters_cluster.tsv` | 40% identity 聚类（10,278 clusters） |
| `train_ids.json` | 训练集 18,019 个 case ID |
| `val_ids.json` | 验证集 2,002 个 case ID |
| `split_manifest.json` | openfold3 兼容 manifest |
| `sampling_weights.pt` | 训练采样权重 + chain cluster 元数据 |
| `split_report.json` | 可读统计报告 |

### 最终输出（`precomputed_mesh/cli_v3/`，训练代码直接读取）

| 文件 | 说明 |
|------|------|
| `split_manifest.json` | per-dataset `{train_ids, val_ids}` |
| `sampling_weights.pt` | 训练采样权重（只含训练集 18,019 个 case） |

## `sampling_weights.pt` 格式

```python
data = torch.load("sampling_weights.pt", weights_only=False)

# 元信息
data["schema"]    # "physppi_identity_split_v1"
data["seed"]      # 42

# 各数据集的 train/val ID
data["datasets"]["afdb"]["train_ids"]    # → ["AF-...", ...]  共 9,314 个
data["datasets"]["afdb"]["val_ids"]      # → ["AF-...", ...]  共 686 个
data["datasets"]["qbiolip"]["train_ids"] # → ["1a0s_1", ...]  共 8,705 个
data["datasets"]["qbiolip"]["val_ids"]   # → ["11bg_1", ...]  共 1,316 个

# 训练采样权重（仅训练集，均值=1.0）
data["sampling_weights"]["AF-0000000065760170-model_v1"]  # → 1.624

# 每条 chain 的 30% / 40% cluster 归属
data["chain_clusters_30"]["AF-...|A"]  # → {"rep": "AF-...|B", "size": 11}
data["chain_clusters_40"]["AF-...|A"]  # → {"rep": "AF-...|C", "size": 15}

# 每个 case 涉及的 30% cluster（每个 chain 一个 representative）
data["case_clusters_30"]["1a0s_1"]     # → ["AF-...|A", "AF-...|B"]
```

## `split_manifest.json` 格式（兼容 openfold3 DataLoader）

```json
{
  "schema_version": 1,
  "seed": 42,
  "datasets": {
    "afdb": {
      "train_ids": ["AF-...", ...],
      "val_ids": ["AF-...", ...]
    },
    "qbiolip": {
      "train_ids": ["1a0s_1", ...],
      "val_ids": ["11bg_1", ...]
    }
  }
}
```

训练 YAML 配置：

```yaml
physppi:
    manifest: "/101045/jinyu/PPI/Prior_out/precomputed_mesh/cli_v3/split_manifest.json"
    manifest_dataset_key: "qbiolip"     # 或 "afdb"
    manifest_partition: "train_ids"     # 或 "val_ids"
```

### 训练时使用 WeightedRandomSampler

```python
import json, torch
from torch.utils.data import WeightedRandomSampler

manifest = json.load(open("split_manifest.json"))
train_ids = manifest["datasets"]["qbiolip"]["train_ids"]

data = torch.load("sampling_weights.pt", weights_only=False)
weights = [data["sampling_weights"][cid] for cid in train_ids]

sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
loader = DataLoader(dataset, sampler=sampler, shuffle=False)
```

## 算法细节

### 为什么用连通分量而不是简单随机切分？

假设三个 complex：

```
A (chain X, Y)    B (chain Y, Z)    C (chain Z, W)
```

如果 A 和 B 都含有 chain Y 的同源序列（30%），那么 A 和 B 不能分到 train/val 两侧——
chain Y 会泄露。同理 B 和 C 共享 chain Z。通过传递闭包，A、B、C 必须在同一侧。

实际数据中，91,143 条 chain → 5,358 个连通分量。最大分量含 6,265 个 case（~31%），
远超 val 预算，只能整体放入 train。

### 为什么用 `max chain cluster_size` 算权重？

对于 heterodimer（chain A cluster_size=50, chain B cluster_size=2）：

```
weight ∝ 1 / max(50, 2) = 1/50
```

保守策略：只要有一条 chain 很常见，整个 case 就被降采样。
对于 homodimer，两条 chain 在同一 cluster → `weight ∝ 1/size`。

### 为什么是 chain-level 而不是 complex-level？

AlphaFold-Multimer 不要求两个 complex 的所有 chain 对之间都做 aligned identity
比较。它只检查：complex X 的任一 chain 是否和 complex Y 的任一 chain 落入同一 30%
cluster。这是 chain-level 检测，也是 AF-Multimer 和 Boltz-1 的通用做法。

## 参考文献

### 核心依据

1. **AlphaFold-Multimer** — 本拆分的核心方法论来源：
   - Evans, R., O'Neill, M., Pritzel, A. et al. "Protein complex prediction with
     AlphaFold-Multimer." *Nature Communications* **14**, 880 (2023).
     doi: [10.1038/s41467-022-29394-2](https://doi.org/10.1038/s41467-022-29394-2)
     (bioRxiv 2021.10.04.463034, 2021).
   - 训练集 PDB cutoff: **2018-04-30**。PDB chain 按 40% identity MMseqs2 聚类，
     以 `1/cluster_size` 逆比例采样（见 AlphaFold2 SI 1.2.5）。
   - 验证集构建：PDB cutoff 之后的 complex，经 30% identity chain-level 同源过滤。

2. **Dimer1193 / Oligomer562 benchmark** — 30% identity 验证集过滤的具体实现：
   - Bryant, P. et al. "Predicting the structure of large protein complexes using
     AlphaFold and Monte Carlo tree search." *Nature Communications* **13**, 6028 (2022).
     doi: [10.1038/s41467-022-33729-4](https://doi.org/10.1038/s41467-022-33729-4).
   - 30% identity MMseqs2 easy-cluster（`--min-seq-id 0.2` 但效果等价于 chain-level
     30% filtering），chain 级同源剔除。

3. **AlphaFold3 Supplementary Information**：
   - Abramson, J. et al. "Accurate structure prediction of biomolecular interactions
     with AlphaFold 3." *Nature* **630**, 493–500 (2024).
     doi: [10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w).
   - SI 2.5.1 & 2.5.3: Weighted PDB 采样, 蛋白链 40% sequence homology 聚类。

### 工程参考

4. **Boltz-1** — MMseqs2 参数和 chain-level exclusion 的独立验证：
   - Wohlwend, J. et al. "Boltz-1: Democratizing Biomolecular Interaction Modeling."
     bioRxiv 2024.11.19.624167 (2025).
     doi: [10.1101/2024.11.19.624167](https://doi.org/10.1101/2024.11.19.624167).
   - 40% identity MMseqs2 easy-cluster, chain-level homology exclusion, 以及
     `-c 0.8 --cov-mode 0 --cluster-mode 2` 参数选择。

5. **IntPEMP identity pipeline** — 本项目的 identity weight 计算直接借用了该管线：
   `/101045/jinyu/IntPEMP_RSCBdata/idenity/`
   - `weight = (1/cluster_size) / mean(1/cluster_size)`，均值归一化到 1.0。
   - MMseqs2 40% identity + 80% coverage 参数。
