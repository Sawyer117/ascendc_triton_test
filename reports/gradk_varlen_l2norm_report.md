# GDN varlen grad_k precision diagnosis

Date: 2026-06-18

## Current Conclusion

The target FLA-NPU varlen case still fails in `grad_k` on the original AscendC wrapper, while the pure Triton reference passes.

The current evidence does **not** prove a standalone `l2norm_bwd` kernel bug. The latest real-context test shows that `l2norm_bwd(normalized_k, k_rstd, dk_core)` matches the PyTorch formula when called directly with the reconstructed real GDN-context tensors.

What is proven:

- Original full AscendC wrapper path fails on `grad_k`.
- Replacing the final q/k l2norm backward step with the PyTorch formula makes the failing target case pass.
- Real standalone `l2norm_bwd(normalized_y, rstd, dy)` passes fixed, varlen, and tail cases.
- Real GDN-context `l2norm_bwd` passes on both q-side and k-side when called directly with the reconstructed `dq_core/dk_core`.

Therefore the remaining localization is:

**The bug is in the original full wrapper/autograd path around the final q/k l2norm backward step, not in the isolated l2norm math that has been reproduced by the standalone and GDN-context tests.**

The next check must compare the original wrapper call path against the reconstructed wrapper/context path, especially saved tensors, tensor layout/stride, call order, mutation, and which exact l2norm function is invoked.

## Target Reproduction Case

```bash
MODE=target_isolated \
VARIANT_LIST="ascendc ascendc_py_l2norm_norm ascendc_py_l2norm_orig triton_full" \
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

Observed replacement result:

| Variant | Result | grad_k max_abs | tail max_abs | Meaning |
| --- | ---: | ---: | ---: | --- |
| `ascendc` | FAIL | 0.0300903 | 0.0300903 | Original AscendC wrapper fails. |
| `ascendc_saved_x` | FAIL | 1.44617 | 0.645447 | Passing original q/k into the old saved-tensor path does not fix it. |
| `ascendc_py_l2norm_norm` | PASS | 0.000976562 | 0.000488281 | Replacing q/k l2norm backward with the PyTorch formula fixes `grad_k`. |
| `ascendc_py_l2norm_orig` | PASS | 0.000976562 | 0.000488281 | Same conclusion under the original-input contract check. |
| `triton_full` | PASS | 0.000488281 | 0.000488281 | Pure Triton remains a valid precision reference/fallback. |

This proves that the final l2norm-backward region is involved. It does not, by itself, prove that the real l2norm kernel arithmetic is wrong.

## Trace Evidence

The component trace ruled out the main GDN core math for the target case:

- Forward GDN output matches the reference.
- `dv_local` matches.
- `dhu` intermediate `dv_mid` matches.
- `dqkwg` outputs (`dq`, `dk`, `dw`, `dg`) match.
- WY outputs (`dk`, `dv`, `dbeta`, `dg`) match.
- Core `dk` sum before final q/k l2norm backward matches.

The trace script has known layout limitations for state tensors such as `fwd.h` and `bwd.dhu.dh`; those shape-mismatch rows should not be used as the final diagnosis.

## Standalone L2Norm Kernel Check

Run:

```bash
bash run_l2norm_kernel_suite.sh
cat ./precision_results/l2norm_kernel/summary.txt
```

This suite calls the real FLA-NPU `l2norm_fwd/l2norm_bwd` directly and compares both possible backward contracts against PyTorch:

- `l2norm_bwd(normalized_y, rstd, dy)`
- `l2norm_bwd(original_x, rstd, dy)`

Observed result:

- `kernel_norm` passes on fixed 1024, fixed 1121, aligned varlen 1024, single-segment varlen 1121, and multi-segment unaligned varlen 1121.
- `kernel_orig` fails on all cases.

So the real FLA-NPU `l2norm_bwd` contract is normalized-output input, not original-x input. The standalone kernel passes under that contract.

## GDN-Context L2Norm Check

Run:

```bash
bash run_l2norm_context_suite.sh
cat ./precision_results/l2norm_context/summary.txt
```

This suite obtains real GDN-produced `dq_core/dk_core`, then compares the final real l2norm kernel call with the PyTorch formula on the same tensors.

Latest observed target result:

| Check | allclose | max_abs | rms | Meaning |
| --- | ---: | ---: | ---: | --- |
| `q_kernel_vs_py_norm` | True | 0.000488281 | 1.72e-05 | q-side real kernel matches PyTorch formula. |
| `k_kernel_vs_py_norm` | True | 0.000488281 | 1.73e-05 | k-side real kernel matches PyTorch formula on real `dk_core`. |
| `q_kernel_vs_ref` | True | 0.000976562 | 4.19e-05 | q-side final gradient matches full PyTorch reference. |
| `k_kernel_vs_ref` | True | 0.000732422 | 4.34e-05 | k-side final gradient matches full PyTorch reference. |
| `tail_k_kernel_vs_ref` | True | 0.000488281 | 3.88e-05 | tail region also passes. |

Additional `dy` controls also pass:

- real `dk_core`
- random `dy` with same RMS
- shuffled flat `dy`
- shuffled token `dy`
- tail-only `dy`
- tail-zero `dy`
- random tail-only `dy` with same RMS

This means the earlier suspected K-side context-kernel failure was a bad intermediate conclusion. Direct real-context l2norm testing now passes.

## Remaining Gap

The only still-failing path is the original full wrapper. Since direct l2norm calls pass, the remaining difference must be in wrapper/autograd integration around the final l2norm backward step. Concrete suspects:

- saved tensor identity: normalized q/k vs original q/k
- tensor layout/stride at the actual wrapper call site
- mutation or aliasing between saved tensors and intermediate tensors
- call order or stream synchronization differences
- calling a different l2norm implementation than the one used in the direct context test
- wrapper-specific reshape/transpose before or after l2norm backward

The next useful test is a wrapper-local A/B:

```bash
MODE=target_isolated \
VARIANT_LIST="ascendc ascendc_kernel_l2norm_norm ascendc_py_l2norm_norm triton_full" \
bash run_gradk_bisect_suite.sh
```

Read it as:

- `ascendc` fails, `ascendc_kernel_l2norm_norm` passes: original wrapper call-site/saved-tensor path differs from the reconstructed normalized-kernel call.
- `ascendc` fails, `ascendc_kernel_l2norm_norm` also fails, `ascendc_py_l2norm_norm` passes: inspect the exact real-kernel invocation inside that wrapper variant.
- `ascendc_py_l2norm_norm` fails: the previous replacement result regressed and must be rechecked first.

## Creative/OpenI Status

The creative-side snapshot passed after the q/k l2norm saved/input contract fix:

```bash
bash run_creative_pair_suite.sh
```

Covered passing cases:

- `fixed_1k_h8_core`
- `fixed_1k_h8_full_l2norm`
- `varlen_single_1024_core`
- `varlen_single_1024_full_l2norm`
- `varlen_aligned_1024_core`
- `varlen_aligned_1024_full_l2norm`
- `varlen_single_1121_core`
- `varlen_single_1121_full_l2norm`

So the tested creative OpenI snapshot is OK after the l2norm contract fix. Do not infer from the current evidence that creative has the same remaining wrapper-path issue unless the actual production creative checkout reproduces it.

## Recommendation

1. Keep `triton_full` as the precision reference.
2. Keep the creative q/k l2norm contract fix.
3. Do not report the current evidence as a standalone K-side l2norm kernel bug.
4. Continue by comparing original wrapper vs reconstructed wrapper-local l2norm call, because standalone and GDN-context l2norm tests both pass now.
