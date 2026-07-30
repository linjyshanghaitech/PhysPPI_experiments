# PDockQHead v1 集成 — 上游文件需要改的地方

## 1. `openfold3/core/model/heads/pdockq.py`

**直接替换为** `4gpu_model_v1/pdockq_head_v1.py` 的内容。

变化：
- 删除 `ChainBalancedInterfacePool`（不再需要 z 的 chain-pair pooling）
- `PDockQHead.__init__` 参数从 `(c_s, c_z, no_bins)` 变为 `(c_s, no_bins)`
- 新增 `self.layer_norm = nn.LayerNorm(c_s)`
- `self.linear` 加 `nn.init.zeros_` 初始化
- `forward()` 签名从 `(si, z, asym_id, token_mask)` 变为 `(si, asym_id, token_mask)`
- `forward()` 内部去掉 z 相关逻辑，在 per-chain mean pool 前加 LayerNorm

## 2. `openfold3/core/model/heads/head_modules.py`

### Line 82-85: 构造 PDockQHead
```python
# 旧:
self.pdockq = PDockQHead(
    c_s=self.config["pdockq"]["c_s"],
    c_z=self.config["pdockq"]["c_z"],
    no_bins=self.config["pdockq"].get("no_bins", 50),
)

# 新:
self.pdockq = PDockQHead(
    c_s=self.config["pdockq"]["c_s"],
    no_bins=self.config["pdockq"].get("no_bins", 50),
)
```

### Line 238-243: 主 forward 调用
```python
# 旧:
aux_out["pdockq_logits"] = self.pdockq(
    si=si.to(device=out_device),
    z=zij.to(device=out_device),
    asym_id=asym_id.to(device=out_device),
    token_mask=token_mask.to(device=out_device),
)

# 新 (去掉 z):
aux_out["pdockq_logits"] = self.pdockq(
    si=si.to(device=out_device),
    asym_id=asym_id.to(device=out_device),
    token_mask=token_mask.to(device=out_device),
)
```

### Line 324-329: `_forward_pdockq_candidate_chunk`
```python
# 旧:
return self.pdockq(
    si=si,
    z=zij,
    asym_id=asym_id,
    token_mask=token_mask,
)

# 新:
return self.pdockq(
    si=si,
    asym_id=asym_id,
    token_mask=token_mask,
)
```

## 3. `openfold3/projects/of3_all_atom/config/model_config.py`

### Line 640-645: pdockq 配置
```python
# 旧:
"pdockq": {
    "c_s": c_s,
    "c_z": c_z,
    "no_bins": 50,
    "candidate_chunk_size": 1,
},

# 新 (去掉 c_z):
"pdockq": {
    "c_s": c_s,
    "no_bins": 50,
    "candidate_chunk_size": 1,
},
```

## 4. `openfold3/tests/test_pdockq_loss.py`

`ChainBalancedInterfacePool` 的测试会失效（该类已被删除）。
需要更新或删除 test_chain_balanced_pool_* 测试。

## 5. `openfold3/core/model/heads/__init__.py` (如果存在)

如果有 `from .pdockq import ChainBalancedInterfacePool, PDockQHead`，改为：
`from .pdockq import PDockQHead`
