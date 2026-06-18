# GDN varlen grad_k precision diagnosis

Date: 2026-06-18

## Summary

The target FLA-NPU varlen case fails in `grad_k` on the original AscendC wrapper, while the pure Triton reference passes. Replacement tests ruled out the GDN core kernels (`dv_local`, `dhu`, `dqkwg`, and WY) as the primary cause: replacing only the final q/k l2norm backward step with a PyTorch formula recovers `grad_k`.

The final isolation result is more specific:

- Standalone real `l2norm_bwd(normalized_y, rstd, dy)` passes on random fixed/varlen/tail inputs.
- In the real GDN backward context, q-side `l2norm_bwd` passes.
- In the real GDN backward context, k-side `l2norm_bwd(normalized_k, k_rstd, dk_core)` fails against the PyTorch formula and against the full PyTorch reference.
- The same `dk_core` input passed through the PyTorch l2norm backward formula matches the full PyTorch reference.

Therefore the current precise diagnosis is: **K-side `l2norm_bwd` fails for the real GDN-produced `dk_core` input on the target varlen case.** This is not a standalone random-input tail failure, and it is not an original-vs-normalized saved-input contract issue for this kernel; the real kernel's working contract is normalized input.

For the creative implementation, the tested `creative_snapshot` path passes after the wrapper fix, but those tests used the local l2norm shim/PyTorch formula path (`mindspeed_triton_shim_used=True`). If a production creative runtime calls the same real FLA-NPU `l2norm_bwd` kernel on GDN-produced `dk_core`, it should be validated with the context test before training.

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

Conclusion: the smallest replacement that makes the failing target case pass is replacing the final q/k `l2norm_bwd` math with the PyTorch formula. The follow-up context test shows the failing side is specifically K-side `l2norm_bwd` when its upstream gradient is the real GDN-produced `dk_core`.

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

## Single-Operator L2Norm Check

A dedicated real-kernel isolation suite has been added for the final validation step:

```bash
bash run_l2norm_kernel_suite.sh
```

This suite calls `l2norm_fwd/l2norm_bwd` directly from `flash-linear-attention-npu/examples/flash_gated_delta_rule.py` and compares both possible backward contracts against PyTorch autograd:

- `l2norm_bwd(normalized_y, rstd, dy)`
- `l2norm_bwd(original_x, rstd, dy)`

It covers fixed 1024, fixed 1121, aligned packed varlen, single-segment 1121 varlen, and multi-segment unaligned varlen cases. The default layout is `BHTD`, matching the FLA-NPU flash wrapper's internal q/k layout. Use `LAYOUTS="BHTD BTHD" bash run_l2norm_kernel_suite.sh` to additionally test the user-facing `BTHD` layout.

Observed standalone result: `kernel_norm` passes on fixed 1024, fixed 1121, aligned varlen 1024, single-segment varlen 1121, and unaligned packed varlen 1121. `kernel_orig` fails on all cases, so the real FLA-NPU `l2norm_bwd` contract is normalized-output input, not original-x input.

## GDN-Context L2Norm Check

The decisive context test is:

```bash
bash run_l2norm_context_suite.sh
```

For target `cu_seqlens=0,1121`, the result was:

| Check | allclose | max_abs | rms | Meaning |
| --- | ---: | ---: | ---: | --- |
| `q_kernel_vs_py_norm` | True | 0.000488281 | 1.72e-05 | q-side real kernel matches PyTorch formula. |
| `k_kernel_vs_py_norm` | False | 0.0198975 | 1.67e-04 | k-side real kernel differs from PyTorch formula on real `dk_core`. |
| `q_kernel_vs_ref` | True | 0.000976562 | 4.19e-05 | q-side final gradient matches full PyTorch reference. |
| `k_kernel_vs_ref` | False | 0.0198975 | 1.72e-04 | k-side final gradient remains wrong. |
| `k_py_norm_vs_ref` | True | 0.000976562 | 4.46e-05 | PyTorch l2norm formula on the same `dk_core` fixes `grad_k`. |

Tail-only `k_kernel_vs_ref` also fails with `max_abs=0.0198975` and `rms=0.000969749`, so the visible error is concentrated more strongly in the 33-token tail.

This is the final narrow localization: **K-side real `l2norm_bwd` fails only after feeding the real GDN-produced `dk_core`; the PyTorch formula on the same inputs passes.**

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

So the creative OpenI branch should be OK for the tested snapshot path after the l2norm contract fix. The remaining caveat is runtime routing: the passing creative tests used the shim/PyTorch l2norm path, not this real FLA-NPU k-side `l2norm_bwd` context. If production creative calls the real kernel, validate it with `run_l2norm_context_suite.sh` or route k-side l2norm backward through the PyTorch/autograd formula until the kernel is fixed.

## Recommendation

1. For creative: keep the fixed q/k l2norm backward contract and run `run_creative_pair_suite.sh` on the actual OpenI checkout before training.
2. For FLA-NPU: report the k-side `l2norm_bwd(normalized_k, k_rstd, dk_core)` precision issue to the kernel owner with the target case above. Include the fact that standalone random-input `kernel_norm` passes, while GDN-context `k_kernel_vs_py_norm` fails.
3. Until the kernel/context issue is fixed, avoid the real k-side `l2norm_bwd` path for varlen GDN. Use the PyTorch formula or a verified fallback.
4. Keep `triton_full` as the precision reference for future regression tests.

