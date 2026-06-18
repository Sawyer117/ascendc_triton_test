# GDN varlen grad_k precision diagnosis

Date: 2026-06-18

## Summary

The target FLA-NPU varlen case fails in `grad_k` on the original AscendC wrapper, while the pure Triton reference passes. The decisive replacement test shows the remaining error is not in the GDN core kernels (`dv_local`, `dhu`, `dqkwg`, or WY). It is in the final q/k `l2norm_bwd` path used after the GDN backward result is produced.

For the creative implementation, the previously identified wrapper issue was the q/k l2norm backward contract. After aligning that path, the tested creative snapshot passes both outer-l2norm and fused-l2norm modes on the covered fixed and varlen cases. The creative repository is therefore OK for the tested GDN path, assuming its runtime uses the fixed l2norm backward contract and does not route back to the faulty FLA-NPU `l2norm_bwd` kernel.

## Reproduction Case

Target case:

```bash
MODE=target_isolated \
VARIANT_LIST="ascendc ascendc_saved_x ascendc_py_l2norm_norm ascendc_py_l2norm_orig triton_full" \
bash run_gradk_bisect_suite.sh
```

Shape and dtype:

```text
case:       varlen
cu_seqlens: 0,1121
heads:      8
key_dim:    128
value_dim:  128
chunk_size: 64
dtype:      bf16
```

## Decisive Result

| Variant | Result | grad_k max_abs | tail max_abs | Meaning |
| --- | ---: | ---: | ---: | --- |
| `ascendc` | FAIL | 0.0300903 | 0.0300903 | Original AscendC wrapper fails. |
| `ascendc_saved_x` | FAIL | 1.44617 | 0.645447 | Passing original q/k into the existing l2norm backward path does not fix it. |
| `ascendc_py_l2norm_norm` | PASS | 0.000976562 | 0.000488281 | Replacing only q/k l2norm backward with PyTorch formula fixes `grad_k`. |
| `ascendc_py_l2norm_orig` | PASS | 0.000976562 | 0.000488281 | Same conclusion under the original-input contract check. |
| `triton_full` | PASS | 0.000488281 | 0.000488281 | Pure Triton remains a valid reference/fallback. |

Conclusion: the smallest replacement that makes the failing target case pass is replacing the final q/k `l2norm_bwd` math with the PyTorch formula. This directly identifies the l2norm backward path as the failing component.

## What This Rules Out

Earlier hybrid replacement runs were noisy because some reconstructed chains had layout or contract mismatches. The final replacement test above avoids that by keeping the AscendC wrapper path intact and changing only the q/k l2norm backward implementation.

Current evidence rules out these components as the primary source of the target `grad_k` error:

- Forward GDN core output: allclose to reference in the trace run.
- `dv_local`: allclose in the trace run.
- `dhu` intermediate `dv_mid`: allclose in the trace run.
- `dqkwg` outputs (`dq`, `dk`, `dw`, `dg`): allclose in the trace run.
- WY outputs (`dk`, `dv`, `dbeta`, `dg`): allclose in the trace run.
- Core `dk` sum before q/k l2norm backward: allclose in the trace run.

The trace script had known layout limitations for state tensors such as `fwd.h` and `bwd.dhu.dh`, so those shape-mismatch rows should not be used as the final diagnosis. The replacement test is the stronger evidence.

## Creative/OpenI Status

The creative-side issue was a wrapper contract mismatch around fused q/k l2norm backward. In `creative_snapshot`, after aligning the saved/input contract, the following suite passed:

```bash
bash run_creative_pair_suite.sh
```

Covered cases:

- `fixed_1k_h8_core`
- `fixed_1k_h8_full_l2norm`
- `varlen_single_1024_core`
- `varlen_single_1024_full_l2norm`
- `varlen_aligned_1024_core`
- `varlen_aligned_1024_full_l2norm`
- `varlen_single_1121_core`
- `varlen_single_1121_full_l2norm`

So the creative OpenI branch should be OK for the tested GDN path after the l2norm contract fix. The remaining caveat is runtime routing: if a deployment path still calls the faulty FLA-NPU `l2norm_bwd` kernel, it should be switched to the fixed wrapper contract, a PyTorch/autograd l2norm backward formula, or the verified Triton fallback.

## Recommendation

1. For creative: keep the fixed q/k l2norm backward contract and run `run_creative_pair_suite.sh` on the actual OpenI checkout before training.
2. For FLA-NPU: report the q/k `l2norm_bwd` precision issue to the kernel owner with the target case above.
3. Until the l2norm backward kernel is fixed, avoid the faulty kernel path for q/k l2norm backward in varlen GDN. Use the PyTorch formula or a verified fallback.
4. Keep `triton_full` as the precision reference for future regression tests.

