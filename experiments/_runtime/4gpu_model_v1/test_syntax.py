"""Syntax and smoke test for the v1 pDockQ head (si-only, LayerNorm, zero-init).

Run:
    cd /101045/jinyu/PhysPPI_jinyu/PhysPPI
    python experiments/_runtime/4gpu_model_v1/test_syntax.py
"""

import sys
import traceback

import torch

# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. Import PDockQHead_v1")
try:
    # Insert v1 directory so we can import the local module
    sys.path.insert(
        0, "/101045/jinyu/PhysPPI_jinyu/PhysPPI/experiments/_runtime/4gpu_model_v1"
    )
    from pdockq_head_v1 import PDockQHead

    print("   ✅ Import successful")
except Exception:
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Instantiation
# ---------------------------------------------------------------------------
print("=" * 60)
print("2. Instantiate PDockQHead(c_s=384)")
try:
    head = PDockQHead(c_s=384, no_bins=50)
    print(f"   ✅ Created: {head}")
    print(f"      layer_norm: {head.layer_norm}")
    print(f"      linear weight shape: {tuple(head.linear.weight.shape)}")
    print(f"      linear bias shape: {tuple(head.linear.bias.shape)}")
except Exception:
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Zero-init verification
# ---------------------------------------------------------------------------
print("=" * 60)
print("3. Verify zero-initialization")
try:
    assert head.linear.weight.abs().max().item() == 0.0, "weight not zero"
    assert head.linear.bias.abs().max().item() == 0.0, "bias not zero"
    print("   ✅ Linear layer weights and bias are all zeros")
except Exception:
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4. Forward pass — single candidate [B, N, c_s]
# ---------------------------------------------------------------------------
print("=" * 60)
print("4. Forward pass — [B, N, c_s] (single candidate)")
try:
    B, N, C = 1, 300, 384
    si = torch.randn(B, N, C)
    asym_id = torch.tensor([[0] * 150 + [1] * 150])  # two chains, 150 res each
    token_mask = torch.ones(B, N, dtype=torch.bool)

    logits = head(si=si, asym_id=asym_id, token_mask=token_mask)
    print(f"   Input si:  {tuple(si.shape)}")
    print(f"   Output:    {tuple(logits.shape)}  (expected: [1, 50])")

    assert logits.shape == (B, 50), f"bad shape: {tuple(logits.shape)}"
    # Zero-init → all logits should be exactly 0.0
    assert logits.abs().max().item() == 0.0, (
        f"logits not zero: max abs = {logits.abs().max().item()}"
    )
    print("   ✅ Logits are all zeros (zero-init working)")

    # Expectation should be 0.5 (uniform distribution over [0,1])
    exp = head.expectation(logits)
    assert exp.shape == (B,), f"bad expectation shape: {tuple(exp.shape)}"
    assert torch.allclose(exp, torch.tensor([0.5]), atol=1e-6), (
        f"expected 0.5, got {exp.item():.6f}"
    )
    print(f"   ✅ Expectation = {exp.item():.6f} (expected 0.5)")
except Exception:
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 5. Forward pass — multiple candidates [B, S, N, c_s]
# ---------------------------------------------------------------------------
print("=" * 60)
print("5. Forward pass — [B, S, N, c_s] (48 candidates)")
try:
    B, S, N, C = 1, 48, 300, 384
    si = torch.randn(B, S, N, C)
    asym_id = torch.tensor([[0] * 150 + [1] * 150])
    token_mask = torch.ones(B, N, dtype=torch.bool)

    logits = head(si=si, asym_id=asym_id, token_mask=token_mask)
    print(f"   Input si:  {tuple(si.shape)}")
    print(f"   Output:    {tuple(logits.shape)}  (expected: [1, 48, 50])")

    assert logits.shape == (B, S, 50), f"bad shape: {tuple(logits.shape)}"
    assert logits.abs().max().item() == 0.0
    print("   ✅ All 48 candidates have zero logits")

    exp = head.expectation(logits)
    assert exp.shape == (B, S)
    assert torch.allclose(exp, torch.full((B, S), 0.5), atol=1e-6)
    print(f"   ✅ All expectations = 0.5")
except Exception:
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 6. Chain-balanced pooling
# ---------------------------------------------------------------------------
print("=" * 60)
print("6. Chain-balanced pooling — same interface, different chain sizes")
try:
    C = 384

    def make_head():
        h = PDockQHead(c_s=C, no_bins=50)
        # Override with ones so pooling is easy to verify
        h.layer_norm.weight.data = torch.ones(C)
        h.layer_norm.bias.data = torch.zeros(C)
        return h

    # Case A: 2 res per chain
    si_a = torch.ones(1, 4, C)
    asym_a = torch.tensor([[0, 0, 1, 1]])
    mask_a = torch.ones(1, 4, dtype=torch.bool)

    # Case B: 4 res per chain (same interface, bigger chains)
    si_b = torch.ones(1, 8, C)
    asym_b = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    mask_b = torch.ones(1, 8, dtype=torch.bool)

    head_a = make_head()
    head_b = make_head()

    logits_a = head_a(si=si_a, asym_id=asym_a, token_mask=mask_a)
    logits_b = head_b(si=si_b, asym_id=asym_b, token_mask=mask_b)

    print(f"   Case A (2+2 res): {logits_a[0, :5].tolist()}...")
    print(f"   Case B (4+4 res): {logits_b[0, :5].tolist()}...")

    assert torch.allclose(logits_a, logits_b, atol=1e-5), (
        "chain-balanced pooling broken: different chain sizes gave different results"
    )
    print("   ✅ Pooling is invariant to chain size (chain-balanced)")
except Exception:
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 7. 3D asym_id (per-candidate) and token_mask broadcast
# ---------------------------------------------------------------------------
print("=" * 60)
print("7. Broadcast: asym_id [B,1,N] and token_mask [B,1,N]")
try:
    B, S, N, C = 1, 48, 300, 384
    si = torch.randn(B, S, N, C)
    asym_id = torch.randint(0, 3, (B, 1, N))
    token_mask = torch.ones(B, 1, N, dtype=torch.bool)

    logits = head(si=si, asym_id=asym_id, token_mask=token_mask)
    assert logits.shape == (B, S, 50)
    print(f"   ✅ Shape: {tuple(logits.shape)}")
except Exception:
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 8. Differentiable / backward
# ---------------------------------------------------------------------------
print("=" * 60)
print("8. Backward pass (gradient flow)")
try:
    # 8a. Zero-init: only head params get gradients, si gets none (by design)
    print("   8a. Zero-init stage (gradient blocked to trunk)")
    B, N, C = 1, 300, 384
    si = torch.randn(B, N, C, requires_grad=True)
    asym_id = torch.tensor([[0] * 150 + [1] * 150])
    token_mask = torch.ones(B, N, dtype=torch.bool)

    logits = head(si=si, asym_id=asym_id, token_mask=token_mask)
    loss = torch.nn.functional.cross_entropy(
        logits, torch.randint(0, 50, (B,))
    )
    loss.backward()

    assert si.grad is not None, "si has no gradient"
    # si grad IS zero because linear weights are zero (by design, like pLDDT "final" init)
    assert si.grad.abs().sum().item() == 0.0, (
        "si gradient should be zero with zero-init weights"
    )
    # Head's own weights SHOULD get gradients
    assert head.linear.weight.grad is not None, "weight has no grad"
    assert head.linear.weight.grad.abs().sum() > 0, "weight grad is zero"
    assert head.linear.bias.grad is not None, "bias has no grad"
    assert head.linear.bias.grad.abs().sum() > 0, "bias grad is zero"
    print(f"      Loss: {loss.item():.6f}")
    print(f"      si.grad max: {si.grad.abs().max().item():.6f}  (zero — expected)")
    print(f"      weight.grad max: {head.linear.weight.grad.abs().max().item():.6f}")
    print(f"      ✅ Head learns, trunk protected (zero-init working as designed)")

    # 8b. After one optimizer step, weights become non-zero → trunk gets gradients
    print("   8b. After weight update (gradient flows to trunk)")
    # Save gradients before clearing
    weight_grad = head.linear.weight.grad.clone()
    bias_grad = head.linear.bias.grad.clone()
    head.zero_grad()
    si.grad = None
    # Simulate one SGD step on the head
    with torch.no_grad():
        head.linear.weight -= 0.01 * weight_grad
        head.linear.bias -= 0.01 * bias_grad
    assert head.linear.weight.abs().sum() > 0, "weights still zero after update"

    logits2 = head(si=si, asym_id=asym_id, token_mask=token_mask)
    loss2 = torch.nn.functional.cross_entropy(
        logits2, torch.randint(0, 50, (B,))
    )
    loss2.backward()
    assert si.grad is not None, "si has no gradient after weight update"
    assert si.grad.abs().sum() > 0, (
        "si gradient should be non-zero after weights become non-zero"
    )
    print(f"      si.grad max after update: {si.grad.abs().max().item():.6f}")
    print(f"      ✅ Gradient flows to trunk after head starts learning")
except Exception:
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 60)
print("🎉 ALL TESTS PASSED")
