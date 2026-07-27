# PhysPPI OpenFold3 Verification

- Targeted PhysPPI tests: 12/12 pass
- Pairformer/diffusion regression tests: 4/4 pass (PairFormer shape + DiffusionConditioning x3)
- OF3 model smoke test: FP32 train/eval pass (kernel tests skipped due to CUDA env)
- Disabled baseline preserved: ✅ `physppi_conditioning is None` when `physppi.enabled = False`

## Spec Compliance Fixes
- Added `nn.LayerNorm(num_rbf + 1)` to `PhysPPIInterfacePairAdapter.proj` (was in plan spec but omitted in initial implementation)
- Updated `test_interface_pair_adapter_updates_z_with_masked_gated_residual` to use non-uniform RBF inputs compatible with LayerNorm

## Non-Negotiable Invariants Verified
1. All adapters respect `token_mask` and `pair_mask` ✅
2. Ground-truth-derived features (source=2) blocked as inference input ✅
3. Interface evidence represented primarily as pair (z) features ✅
4. Gated residuals initialized near zero (gate_init=-4.0) ✅
5. Protein residue tokens are the supported scope ✅
