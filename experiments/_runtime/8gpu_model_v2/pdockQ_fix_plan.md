# pDockQ / DockQ Residue Reindex 实施计划

> **给后续 agent / 开发者：** 实施本计划时必须按任务逐项推进。推荐使用 `superpowers:subagent-driven-development`，或使用 `superpowers:executing-plans` 在当前会话内执行。所有步骤使用 `- [ ]` checklist 语法，便于跟踪。

**目标：** 修复 PhysPPI pDockQ 训练 target 构造，使 DockQ 不再把 PDB 原始 `residue_index` 当作 scatter/gather 的 residue slot。

**总体架构：** 保留 `batch["residue_index"]` 原样，继续供模型相对位置特征、审计和其他逻辑使用。只在 pDockQ 的 DockQ target 构造入口生成一份 DockQ 专用的、0-based、按链紧凑的 residue slot，再 broadcast 到 atom 维度后传给 `quality.dockq()`。同时删除 runner 层“DockQ mapping invalid 就 mask target”的临时 fallback，让真正的 index bug 直接暴露。

**技术栈：** Python 3.10、PyTorch、pytest、OpenFold3/PhysPPI training target path。

---

## 已确认的范围和语义

本次只修 pDockQ training target 路径：

- 修改 `openfold3/core/loss/pdockq.py`
- 修改 `openfold3/projects/of3_all_atom/runner.py`
- 修改 pDockQ 相关测试

本次不修改：

- `openfold3/core/metrics/quality.py` 中 full-complex DockQ metric 路径
- 数据集里的原始 `residue_index`
- 模型特征中的 `batch["residue_index"]`

必须保留：

- `openfold3/core/utils/atomize_utils.py` 里的 bounds check
- `openfold3/tests/test_atomize_utils.py` 里的 out-of-range 检查测试

DockQ 专用 slot helper 只使用以下 batch 字段：

- `token_mask`
- `num_atoms_per_token`
- `asym_id`
- `residue_index`

有效 token 定义：

```python
token_mask.bool() & (num_atoms_per_token > 0) & (asym_id > 0)
```

slot 生成语义：

- 按模型 token 顺序遍历。
- 每条 `asym_id` 独立维护状态。
- 如果同一条 chain 的当前有效 token 和该 chain 上一个有效 token 的 `residue_index` 相同，则复用当前 slot。
- 如果同一条 chain 进入新的 `residue_index`，则分配该 chain 的下一个 0-based slot。
- 输出 slot 是每条 chain 内 0-based 且紧凑的编号。
- padding / invalid token 可以保持 slot `0`，因为它们不会通过 `num_atoms_per_token` 贡献 atom。

这个语义既解决负 residue id、gap residue id、PDB 从负数开始编号的问题，也保留 atomized / modified residue 中相邻同 `residue_index` token 的 residue-level grouping。

---

## 文件职责

- `openfold3/core/loss/pdockq.py`
  - 新增 `_dockq_residue_slots_from_batch(batch)`。
  - 在 `per_candidate_complex_dockq()` 中用 helper 输出替代 `atomize(batch["residue_index"])`。

- `openfold3/projects/of3_all_atom/runner.py`
  - 删除 `_populate_physppi_pdockq_targets()` 里捕获 DockQ mapping `ValueError` 并 mask target 的逻辑。
  - 让 `per_candidate_complex_dockq()` 的 mapping error 正常抛出。

- `openfold3/tests/test_pdockq_loss.py`
  - 更新现有 fake DockQ test 的 `res_id_atomized` 期望。
  - 新增 negative / interleaved / padding residue-slot regression test。

- `openfold3/tests/projects/of3_all_atom/physppi/model/test_pdockq_training_target.py`
  - 删除旧的“invalid DockQ mapping 时 mask target”测试。
  - 新增“负原始 residue id 不导致 target mask”的 runner 层测试。
  - 新增“DockQ mapping error 会传播”的 runner 层测试。

- `openfold3/tests/test_atomize_utils.py`
  - 保留现有 out-of-range bounds check 测试。

---

## Task 1: 添加 pDockQ residue slot 红测

**文件：**

- 修改：`openfold3/tests/test_pdockq_loss.py`

- [ ] **Step 1: 更新现有 fake DockQ 期望**

在 `test_per_candidate_dockq_derives_atomized_inputs_from_training_batch` 里，将：

```python
assert captured["res_id_atomized"].tolist() == [
    [[10, 10, 20, 20], [10, 10, 20, 20]]
]
```

改为：

```python
assert captured["res_id_atomized"].tolist() == [
    [[0, 0, 0, 0], [0, 0, 0, 0]]
]
```

原因：两个 token 属于不同 chain，每条 chain 的第一个 residue slot 都应为 `0`。

- [ ] **Step 2: 新增负 residue、chain interleaving、padding 的回归测试**

在同一文件中追加：

```python
def test_per_candidate_dockq_reindexes_negative_interleaved_and_padded_residues(
    monkeypatch,
):
    import openfold3.core.loss.pdockq as module

    captured = {}

    def fake_dockq(**kwargs):
        captured.update(kwargs)
        return type(
            "Result",
            (),
            {
                "chain_pair_to_dockq": {
                    (1, 2, 0): torch.tensor([0.9]),
                }
            },
        )()

    monkeypatch.setattr(module, "dockq", fake_dockq)

    prediction = torch.randn(1, 1, 7, 3, dtype=torch.bfloat16, requires_grad=True)
    batch = {
        "atom_mask": torch.ones(1, 1, 7, dtype=torch.bool),
        "token_mask": torch.tensor([[[1, 1, 1, 1, 1, 1, 1]]], dtype=torch.float32),
        "num_atoms_per_token": torch.tensor([[[1, 1, 2, 1, 1, 1, 0]]]),
        "asym_id": torch.tensor([[[1, 1, 1, 2, 2, 1, 0]]]),
        "residue_index": torch.tensor([[[-4, -4, -3, 1, 1, 2, 99]]]),
        "ref_atom_name_chars": torch.zeros(1, 1, 7, 4, 64),
        "is_protein": torch.ones(1, 1, 7, dtype=torch.bool),
        "is_rna": torch.zeros(1, 1, 7, dtype=torch.bool),
        "is_dna": torch.zeros(1, 1, 7, dtype=torch.bool),
        "is_atomized": torch.zeros(1, 1, 7, dtype=torch.bool),
        "ground_truth": {
            "atom_resolved_mask": torch.ones(1, 1, 7, dtype=torch.bool),
            "inter_filter_atomized": torch.ones(1, 1, 7, 7, dtype=torch.bool),
        },
    }

    score, valid = per_candidate_complex_dockq(
        prediction=prediction,
        gt=torch.zeros(1, 1, 7, 3, dtype=torch.bfloat16),
        batch=batch,
    )

    assert captured["res_id_atomized"].tolist() == [[[0, 0, 1, 1, 0, 0, 2]]]
    assert min(captured["res_id_atomized"].reshape(-1).tolist()) >= 0
    torch.testing.assert_close(score, torch.tensor([[0.9]]))
    assert valid.tolist() == [[True]]
```

期望 slot 解释：

- chain 1 的 residue `-4` 跨两个相邻 token，slot 复用 `0`。
- chain 1 的 residue `-3` 使用 slot `1`，并 broadcast 到两个 atom。
- chain 2 的 residue `1` 跨两个相邻 token，使用该 chain 的 slot `0`。
- token 顺序返回 chain 1 后，residue `2` 使用 chain 1 的 slot `2`。
- 最后一个 padding token `num_atoms_per_token == 0`，不贡献 atom。

- [ ] **Step 3: 运行红测，确认实现前失败**

```bash
LD_LIBRARY_PATH=/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/cuequivariance_ops/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cublas/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib \
PYTHONPATH=/101045/jinyu/PhysPPI_jinyu/PhysPPI:${PYTHONPATH:-} \
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python -m pytest \
  openfold3/tests/test_pdockq_loss.py::test_per_candidate_dockq_derives_atomized_inputs_from_training_batch \
  openfold3/tests/test_pdockq_loss.py::test_per_candidate_dockq_reindexes_negative_interleaved_and_padded_residues \
  -q
```

实现前预期：

```text
FAILED openfold3/tests/test_pdockq_loss.py::test_per_candidate_dockq_derives_atomized_inputs_from_training_batch
FAILED openfold3/tests/test_pdockq_loss.py::test_per_candidate_dockq_reindexes_negative_interleaved_and_padded_residues
```

---

## Task 2: 实现 pDockQ-local DockQ residue slot helper

**文件：**

- 修改：`openfold3/core/loss/pdockq.py`
- 测试：`openfold3/tests/test_pdockq_loss.py`

- [ ] **Step 1: 在 `dockq_to_soft_bins()` 后新增 helper**

```python
def _dockq_residue_slots_from_batch(batch: dict) -> torch.Tensor:
    """Build DockQ-private compact residue slots from token order.

    Raw PDB residue IDs can be negative or sparse and are not safe scatter/gather
    slots. DockQ only needs residue grouping, so this allocates per-chain
    0-based slots while preserving adjacent same-residue grouping within each
    chain.
    """

    token_mask = batch["token_mask"].bool()
    num_atoms_per_token = batch["num_atoms_per_token"].to(device=token_mask.device)
    asym_id = batch["asym_id"].to(device=token_mask.device)
    residue_index = batch["residue_index"].to(device=token_mask.device)

    expected_shape = token_mask.shape
    for name, value in (
        ("num_atoms_per_token", num_atoms_per_token),
        ("asym_id", asym_id),
        ("residue_index", residue_index),
    ):
        if value.shape != expected_shape:
            raise ValueError(
                f"{name} must match token_mask shape, got "
                f"{tuple(value.shape)} and {tuple(expected_shape)}"
            )

    effective_token = (
        token_mask & (num_atoms_per_token.long() > 0) & (asym_id.long() > 0)
    )
    slots = torch.zeros_like(residue_index, dtype=torch.long, device=token_mask.device)
    if slots.numel() == 0:
        return slots

    n_token = int(token_mask.shape[-1])
    flat_effective = effective_token.reshape(-1, n_token)
    flat_asym_id = asym_id.long().reshape(-1, n_token)
    flat_residue_index = residue_index.long().reshape(-1, n_token)
    flat_slots = slots.reshape(-1, n_token)

    for row in range(flat_slots.shape[0]):
        chain_state: dict[int, tuple[int, int, int]] = {}
        for token_pos in range(n_token):
            if not bool(flat_effective[row, token_pos].item()):
                continue
            chain_id = int(flat_asym_id[row, token_pos].item())
            residue_id = int(flat_residue_index[row, token_pos].item())
            state = chain_state.get(chain_id)
            if state is not None and state[0] == residue_id:
                slot = state[1]
                next_slot = state[2]
            else:
                slot = state[2] if state is not None else 0
                next_slot = slot + 1
            chain_state[chain_id] = (residue_id, slot, next_slot)
            flat_slots[row, token_pos] = slot

    return slots
```

- [ ] **Step 2: 在 `per_candidate_complex_dockq()` 中使用 helper**

在局部 `atomize()` helper 后插入：

```python
dockq_residue_slots = _dockq_residue_slots_from_batch(batch)
```

然后把：

```python
"res_id_atomized": expand_candidates(
    atomize(batch["residue_index"]), "res_id_atomized"
),
```

替换为：

```python
"res_id_atomized": expand_candidates(
    atomize(dockq_residue_slots), "res_id_atomized"
),
```

- [ ] **Step 3: 运行 pDockQ loss 测试**

```bash
LD_LIBRARY_PATH=/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/cuequivariance_ops/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cublas/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib \
PYTHONPATH=/101045/jinyu/PhysPPI_jinyu/PhysPPI:${PYTHONPATH:-} \
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python -m pytest \
  openfold3/tests/test_pdockq_loss.py -q
```

实现后预期：

```text
passed
```

---

## Task 3: 删除 runner 层 DockQ mapping mask fallback

**文件：**

- 修改：`openfold3/projects/of3_all_atom/runner.py`
- 修改：`openfold3/tests/projects/of3_all_atom/physppi/model/test_pdockq_training_target.py`

- [ ] **Step 1: 用错误传播测试替换旧 mask 测试**

```python
def test_runner_propagates_pdockq_mapping_errors(monkeypatch):
    import openfold3.projects.of3_all_atom.runner as module

    def fake_per_candidate_complex_dockq(*, prediction, gt, batch):
        raise ValueError(
            "atom_to_token_index_list[0] contains indices outside [0, 4)"
        )

    monkeypatch.setattr(
        module, "per_candidate_complex_dockq", fake_per_candidate_complex_dockq
    )
    batch = _minimal_target_batch()
    batch["pdb_id"] = ["7k3h_1"]
    outputs = {
        "atom_positions_diffusion": torch.randn(1, 2, 4, 3),
        "pdockq_logits": torch.randn(1, 2, 50),
    }

    with pytest.raises(ValueError, match="atom_to_token_index_list"):
        OpenFold3AllAtom._populate_physppi_pdockq_targets(
            _runner(PHYSPPI_SCOPE), batch, outputs
        )

    assert "physppi_pdockq_target" not in batch
    assert "physppi_pdockq_valid" not in batch
    assert "physppi_pdockq_target_score" not in batch
```

- [ ] **Step 2: 新增负原始 residue id 不被 mask 的 runner 测试**

```python
def test_runner_does_not_mask_pdockq_targets_for_negative_original_residue_ids(
    monkeypatch,
    caplog,
):
    import openfold3.projects.of3_all_atom.runner as module

    def fake_per_candidate_complex_dockq(*, prediction, gt, batch):
        assert batch["residue_index"].tolist() == [[[-4, -3, 1, 2]]]
        return torch.tensor([[0.25, 0.75]]), torch.tensor([[True, True]])

    monkeypatch.setattr(
        module, "per_candidate_complex_dockq", fake_per_candidate_complex_dockq
    )
    batch = _minimal_target_batch()
    batch.update(
        {
            "pdb_id": ["7k3h_1"],
            "token_mask": torch.ones(1, 1, 4),
            "num_atoms_per_token": torch.ones(1, 1, 4, dtype=torch.long),
            "asym_id": torch.tensor([[[1, 1, 2, 2]]]),
            "residue_index": torch.tensor([[[-4, -3, 1, 2]]]),
            "is_protein": torch.ones(1, 1, 4, dtype=torch.bool),
            "is_rna": torch.zeros(1, 1, 4, dtype=torch.bool),
            "is_dna": torch.zeros(1, 1, 4, dtype=torch.bool),
            "is_atomized": torch.zeros(1, 1, 4, dtype=torch.bool),
        }
    )
    outputs = {
        "atom_positions_diffusion": torch.randn(1, 2, 4, 3),
        "pdockq_logits": torch.randn(1, 2, 50),
    }

    OpenFold3AllAtom._populate_physppi_pdockq_targets(
        _runner(PHYSPPI_SCOPE), batch, outputs
    )

    assert batch["physppi_pdockq_valid"].tolist() == [[True, True]]
    assert batch["physppi_pdockq_diag_valid_count"] == 2
    assert batch["physppi_pdockq_target"].sum().item() > 0
    assert "Skipping pDockQ target" not in caplog.text
```

- [ ] **Step 3: 删除 `_populate_physppi_pdockq_targets()` 中的 `try/except ValueError` fallback**

把捕获 `atom_to_token_index_list outside` 并构造全零 target / valid 的分支删除，保留直接调用：

```python
with torch.no_grad():
    scores, valid = per_candidate_complex_dockq(
        prediction=prediction,
        gt=ground_truth["atom_positions"],
        batch=batch,
    )
    if num_bins == 50:
        target = dockq_to_soft_bins(scores, no_bins=50)
        target = target * valid.unsqueeze(-1).to(dtype=target.dtype)
    else:
        target = scores.unsqueeze(-1)
```

- [ ] **Step 4: 运行 runner pDockQ target 测试**

```bash
LD_LIBRARY_PATH=/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/cuequivariance_ops/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cublas/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib \
PYTHONPATH=/101045/jinyu/PhysPPI_jinyu/PhysPPI:${PYTHONPATH:-} \
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python -m pytest \
  openfold3/tests/projects/of3_all_atom/physppi/model/test_pdockq_training_target.py -q
```

预期：

```text
passed
```

---

## Task 4: 保留底层 atomize bounds check

**文件：**

- 检查：`openfold3/core/utils/atomize_utils.py`
- 检查：`openfold3/tests/test_atomize_utils.py`

- [ ] **Step 1: 确认测试仍存在**

`openfold3/tests/test_atomize_utils.py` 中应保留：

```python
def test_aggregate_atom_feat_to_tokens_nd_rejects_out_of_range_token_indices():
    atom_feat = torch.ones(2)
    atom_to_token_index = torch.tensor([0, 2])
    token_mask = torch.ones(2, dtype=torch.bool)

    with pytest.raises(ValueError, match="atom_to_token_index_list\\[0\\].*outside"):
        aggregate_atom_feat_to_tokens_nd(
            atom_feat,
            atom_dims=[0],
            atom_to_token_index_list=[atom_to_token_index],
            token_sizes=[2],
            token_masks=[token_mask],
        )
```

- [ ] **Step 2: 运行该测试**

```bash
LD_LIBRARY_PATH=/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/cuequivariance_ops/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cublas/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib \
PYTHONPATH=/101045/jinyu/PhysPPI_jinyu/PhysPPI:${PYTHONPATH:-} \
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python -m pytest \
  openfold3/tests/test_atomize_utils.py::test_aggregate_atom_feat_to_tokens_nd_rejects_out_of_range_token_indices -q
```

预期：

```text
passed
```

---

## Task 5: 聚焦验证

- [ ] **Step 1: 运行聚焦 pytest 集合**

```bash
LD_LIBRARY_PATH=/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/cuequivariance_ops/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cublas/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib \
PYTHONPATH=/101045/jinyu/PhysPPI_jinyu/PhysPPI:${PYTHONPATH:-} \
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python -m pytest \
  openfold3/tests/test_pdockq_loss.py \
  openfold3/tests/test_atomize_utils.py \
  openfold3/tests/projects/of3_all_atom/physppi/model/test_pdockq_training_target.py \
  -q
```

预期：

```text
passed
```

- [ ] **Step 2: 运行语法检查**

```bash
PYTHONPATH=/101045/jinyu/PhysPPI_jinyu/PhysPPI:${PYTHONPATH:-} \
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python -m py_compile \
  openfold3/core/loss/pdockq.py \
  openfold3/projects/of3_all_atom/runner.py \
  openfold3/core/utils/atomize_utils.py
```

预期：exit code `0`。

- [ ] **Step 3: 运行 whitespace / conflict marker 检查**

```bash
git diff --check -- \
  openfold3/core/loss/pdockq.py \
  openfold3/projects/of3_all_atom/runner.py \
  openfold3/core/utils/atomize_utils.py \
  openfold3/tests/test_pdockq_loss.py \
  openfold3/tests/test_atomize_utils.py \
  openfold3/tests/projects/of3_all_atom/physppi/model/test_pdockq_training_target.py
```

预期：exit code `0`。

---

## Task 6: 固定 trace 的 `7k3h_1` 真实 smoke

**事故现场 trace：**

```text
sample_id=7k3h_1
dataset=qbiolip_train
epoch=0
global_slot=254
rank=6
micro_step=31
requested_level=3
contact_cutoff=13.518930778381542
occurrence_seed=17155105189726813656
```

- [ ] **Step 1: 用真实样本构建 pDockQ target**

运行固定 trace smoke，要求：

- `occurrence_seed == 17155105189726813656`
- `pdockq_valid_count > 0`
- `pdockq_target_sum > 0`
- 不再打印 `Skipping pDockQ target`

完整 smoke 脚本：

```bash
LD_LIBRARY_PATH=/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/cuequivariance_ops/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cublas/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/101045/jinyu/.local/share/mamba/envs/PhysPPI/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib \
PYTHONPATH=/101045/jinyu/PhysPPI_jinyu/PhysPPI:${PYTHONPATH:-} \
/101045/jinyu/.local/share/mamba/envs/PhysPPI/bin/python - <<'PY'
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from openfold3.core.data.framework.data_module import openfold_batch_collator
from openfold3.core.data.framework.single_datasets.qbiolip10k import QBioLiP10kDataset
from openfold3.core.data.framework.stochastic_sampler_dataset import SampleOccurrence
from openfold3.projects.of3_all_atom.config.dataset_configs import QBioLiP10kConfig
from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom

yaml_path = Path("/101045/jinyu/PhysPPI_jinyu/PhysPPI/experiments/_runtime/8gpu_model_v2.yaml")
cfg = yaml.safe_load(yaml_path.read_text())
spec = cfg["dataset_configs"]["train"]["qbiolip_train"]
dataset_config = dict(spec["config"])
dataset_config["name"] = "qbiolip_train"
dataset_config["dataset_paths"] = cfg["dataset_paths"]["qbiolip_train"]

dataset = QBioLiP10kDataset(QBioLiP10kConfig.model_validate(dataset_config))
matches = dataset.datapoint_cache.index[
    dataset.datapoint_cache["pdb_id"] == "7k3h_1"
].tolist()
if len(matches) != 1:
    raise SystemExit(f"expected exactly one 7k3h_1 datapoint, got {matches}")

index = int(matches[0])
dataset.set_current_datapoint_index(index)
dataset.set_current_occurrence(
    SampleOccurrence(
        dataset_idx=0,
        datapoint_idx=index,
        epoch=0,
        global_slot=254,
    )
)

with dataset.audit_prior_view(
    requested_level=3,
    contact_cutoff=13.518930778381542,
):
    features = dataset[index]

seed = int(features["physppi_occurrence_seed"].item())
if seed != 17155105189726813656:
    raise SystemExit(f"unexpected occurrence seed {seed}")

batch = openfold_batch_collator([features])

def add_candidate_axis(value):
    if isinstance(value, dict):
        return {key: add_candidate_axis(child) for key, child in value.items()}
    if torch.is_tensor(value) and value.ndim >= 2 and value.shape[0] == 1:
        return value.unsqueeze(1)
    return value

batch = add_candidate_axis(batch)
gt = batch["ground_truth"]["atom_positions"]
prediction = gt.expand(gt.shape[0], 2, gt.shape[-2], gt.shape[-1]).clone()
outputs = {
    "atom_positions_diffusion": prediction,
    "pdockq_logits": torch.zeros(1, 2, 50),
}
runner = SimpleNamespace(
    config=SimpleNamespace(
        settings=SimpleNamespace(training_scope="physppi_lora_assessment"),
    )
)

OpenFold3AllAtom._populate_physppi_pdockq_targets(runner, batch, outputs)
valid_count = int(batch["physppi_pdockq_diag_valid_count"])
target_sum = float(batch["physppi_pdockq_target"].sum())
score_sum = float(batch["physppi_pdockq_target_score"].sum())
print("occurrence_seed", seed)
print("pdockq_valid_count", valid_count)
print("pdockq_target_sum", target_sum)
print("pdockq_target_score_sum", score_sum)
if valid_count <= 0:
    raise SystemExit("expected pdockq_valid_count > 0")
if target_sum <= 0.0:
    raise SystemExit("expected nonzero pDockQ target")
PY
```

预期至少包含：

```text
occurrence_seed 17155105189726813656
pdockq_valid_count 2
```

- [ ] **Step 2: 确认 runner 中没有旧 fallback 文本**

```bash
rg -n "Skipping pDockQ target|DockQ atom-to-token mapping is invalid" \
  openfold3/projects/of3_all_atom/runner.py
```

预期：无输出。

---

## 最终验收清单

- [ ] `openfold3/tests/test_pdockq_loss.py` 通过。
- [ ] `openfold3/tests/test_atomize_utils.py` 通过。
- [ ] `openfold3/tests/projects/of3_all_atom/physppi/model/test_pdockq_training_target.py` 通过。
- [ ] `py_compile` 对 `pdockq.py`、`runner.py`、`atomize_utils.py` 返回 exit code `0`。
- [ ] `git diff --check` 对相关文件返回 exit code `0`。
- [ ] 固定 trace 的 `7k3h_1` smoke 报告 `pdockq_valid_count > 0`。
- [ ] `runner.py` 不再包含 `Skipping pDockQ target` fallback。
- [ ] 不删除 `experiments/_runtime/8gpu_model_v2` 下的旧日志、旧 metrics 和旧 evidence。

---

## 已执行验证记录

本计划已经在当前 dirty `main` checkout 中执行过一次。验证结果如下：

```text
53 passed, 16 warnings
```

固定 trace smoke 输出：

```text
occurrence_seed 17155105189726813656
pdockq_valid_count 2
pdockq_target_sum 2.0
pdockq_target_score_sum 2.0
```

`py_compile`：exit code `0`。

`git diff --check`：exit code `0`。

因为当时工作区已有大量未提交改动，且包括本次触碰文件，所以没有按计划做中间 commit，避免把无关改动混进提交。

---

## 暂缓的后续问题

`openfold3/core/metrics/quality.py` 中 full-complex DockQ metric 路径仍然会在 `dockq_full_complex()` 前 atomize 原始 `batch["residue_index"]`。这次修复只覆盖 pDockQ training target，不覆盖该 metric 路径。

如果后续该路径也遇到负 residue id 或稀疏/gapped residue numbering，应把 slot helper 抽到共享 utility，再同时接入 pDockQ 和 full-complex DockQ。
