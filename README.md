# AscendC GDN Precision Tests

This repo now has two precision-test layers:

- `compare_creative_gdn_pair.py`: compares the creative repo's pure Triton GDN against its AscendC-mixed GDN. Use this first for the original creative-code question.
- `compare_gdn_precision.py`: compares the FLA-npu example AscendC wrapper against an embedded PyTorch reference. Use this to characterize the installed AscendC custom ops after the creative path is understood.

Both layers check forward output plus gradients of `q`, `k`, `v`, `beta`, and `g`, and both support fixed length plus packed varlen via `cu_seqlens`.

The AscendC-mixed paths still need the FLA-npu custom op package installed, because both creative and FLA-npu ultimately call `torch.ops.npu.*` custom ops.


## Creative Pair Test

For the original creative-code question, use `compare_creative_gdn_pair.py` or `run_creative_pair_suite.sh` first. This compares a vendored snapshot of the two creative implementations:

- Pure Triton baseline: `mindspeed_mm/fsdp/models/qwen3_5/chunk_gated_delta_rule.py`
- AscendC mixed path: `mindspeed_mm/fsdp/models/qwen3_5/flash_gated_delta_rule.py`

This is different from the older FLA-npu/PyTorch-reference test below. The FLA-npu test is still useful for characterizing the installed AscendC custom ops, but it is not a substitute for creative-vs-creative validation.

Run from the test repo root:

```bash
cd /home/canada_group_account/a00652497/bytedance/ascendc_triton_test
export LD_LIBRARY_PATH=/home/canada_group_account/CANN/9.0.0.0430/cann-9.0.0/opp/vendors/fla_npu_transformer/op_api/lib/:${LD_LIBRARY_PATH}
bash run_creative_pair_suite.sh
```

By default this uses `./creative_snapshot` and runs two layers for each safe case: `*_core` pre-normalizes q/k once in the test harness and disables wrapper q/k l2norm to isolate the GDN operator chain, while `*_full_l2norm` feeds raw q/k and lets the pure and mixed wrappers each use their own q/k l2norm path. This keeps core-GDN and full-wrapper conclusions separate.

To validate a newer creative branch instead of the vendored snapshot, override it explicitly:

```bash
CREATIVE_REPO=/path/to/qwen3.5_omni_creative bash run_creative_pair_suite.sh
```

The suite writes to `./creative_pair_results/` and compares `output`, `grad_q`, `grad_k`, `grad_v`, `grad_beta`, and `grad_g`. The default cases are fixed-length controls, aligned varlen controls, and the single-segment non-64-aligned packed-tail case `cu_seqlens=0,1121`.

Optional modes:

```bash
# Skip the full-wrapper l2norm layer and run only pre-normalized core-GDN comparisons.
RUN_FULL_L2NORM=0 bash run_creative_pair_suite.sh

# Run multi-segment unaligned cases that can currently trigger an AICore exception.
RUN_UNSAFE=1 bash run_creative_pair_suite.sh
```

Run one focused case:

```bash
python compare_creative_gdn_pair.py \
  --case varlen \
  --cu-seqlens 0,1121 \
  --heads 8 \
  --key-dim 128 \
  --value-dim 128 \
  --chunk-size 64 \
  --device 0 \
  --no-qk-l2norm \
  --pre-normalize-qk \
  --output-json ./creative_pair_results/varlen_single_1121.json
```

The focused command above is the core-GDN variant. Omit `--no-qk-l2norm` to run the full-wrapper variant where the pure and mixed implementations each use their own q/k l2norm path. Pass `--creative-repo /path/to/qwen3.5_omni_creative` only when you intentionally want to test an external checkout.

If the creative mixed path imports `mindspeed.lite.ops.triton.*` but that external `mindspeed` package is absent, the script shims those helper imports to the snapshot's local `mindspeed_mm/fsdp/models/qwen3_5/triton/*` modules and records `mindspeed_triton_shim_used=true` in JSON. Use `--no-mindspeed-triton-shim` to require the exact external import path.

Interpretation:

- If creative pure Triton vs creative AscendC mixed is worse than the FLA-npu/PyTorch-reference result, report the creative-wrapper mismatch first.
- If creative mixed is worse, the next engineering step is to port the cleaner FLA-npu wrapper logic into creative, then re-run the creative pair test. Do not make that code change in this test repo.
- If creative pair matches the FLA-npu behavior, use the FLA-npu/PyTorch-reference scripts to quantify the installed AscendC op issue and hand the failing shapes/metrics to the kernel owner.


## L2Norm Mode Suite

Use this when you need to prove whether the q/k l2norm path is the only reason full-wrapper precision differs. It runs the same creative-vs-creative GDN cases twice:

- `outer`: fixed path. The mixed flash wrapper applies the same Python/autograd l2norm as the Triton baseline before entering the custom autograd Function.
- `kernel`: old path. The mixed flash wrapper leaves `use_qk_l2norm_in_kernel=True`, so the custom Function uses `mindspeed.lite.ops.triton.l2norm_fwd/bwd`.

It also runs a standalone l2norm check against Python autograd for the same shapes. By default the standalone test tries to import the real external `mindspeed.lite.ops.triton.l2norm`; if that package is absent it reports `skipped=True` instead of pretending the local shim is a real kernel. Use `L2NORM_SOURCE=shim` only when you explicitly want to test the fallback shim contract.

```bash
bash run_l2norm_mode_suite.sh

# Optional: test the local shim instead of the real external l2norm package.
L2NORM_SOURCE=shim bash run_l2norm_mode_suite.sh
```

Read `./l2norm_mode_results/summary.txt`:

- If all `outer` GDN rows pass and `kernel` fails, the remaining mismatch is in the l2norm path rather than the GDN core.
- If `kernel` fails for fixed/aligned cases too, then the problem is not limited to non-64-aligned varlen.
- In standalone l2norm rows, `original_input_ok` versus `normalized_output_ok` shows which backward contract the available `l2norm_bwd` actually implements. The old flash path passes the normalized output into `l2norm_bwd`, so `normalized_output_ok=False` directly explains q/k gradient failure.

## Expected Environment

Run on the Ascend server with the same Python environment that has working `torch`, `torch_npu`, and `fla_npu`.

Known-good family from the FLA-npu notes:

- CANN 8.5+ or 9.x
- Ascend PyTorch release `v26.1.0-beta.1`
- `torch==2.7.1`
- `torch_npu==2.7.1.post5`, matching your Python ABI and CPU arch
- `triton-ascend==3.2.0`, `pybind11`, and `flash-linear-attention-npu/requirements.txt`

If `torch_custom/fla_npu/build.sh` reports `No module named torchnpugen`, your `torch_npu` wheel is missing or from the wrong release family. The matching `torch_npu` wheel should provide `torchnpugen`.

## Install Triton-Ascend

For this FLA-npu GDN stack, follow the FLA-npu README rather than vLLM/Speculators deployment guides:

- Use `triton-ascend==3.2.0`.
- Do not keep community `triton` installed in the same environment.
- `triton-ascend==3.2.1` belongs to the newer CANN 9.0/vLLM stack and can expose a different `triton.language` API surface for these helper kernels.

Clean a previously mixed install first:

```bash
python -m pip uninstall -y triton triton-ascend
python -m pip uninstall -y triton triton-ascend
python - <<'PY'
import importlib.util
spec = importlib.util.find_spec("triton")
print("triton spec after uninstall:", None if spec is None else spec.origin)
raise SystemExit(0 if spec is None else 1)
PY
```

If the last command still prints a `triton` path, remove only the leftover paths under this conda env, then install again:

```bash
python - <<'PY'
import pathlib
import shutil
import site
import sys
import sysconfig

roots = set(site.getsitepackages())
roots.add(sysconfig.get_paths()["purelib"])
roots.add(sysconfig.get_paths()["platlib"])
for root in sorted(pathlib.Path(x) for x in roots):
    if not str(root).startswith(sys.prefix) or not root.exists():
        continue
    for pattern in ("triton", "triton-*.dist-info", "triton_ascend-*.dist-info"):
        for path in root.glob(pattern):
            print("remove", path)
            shutil.rmtree(path)
PY
```

Install the FLA-npu expected package:

```bash
python -m pip install --no-cache-dir --no-deps triton-ascend==3.2.0
```

Verify that the imported `triton` package has the APIs used by FLA-npu:

```bash
python - <<'PY'
import triton
import triton.language as tl
from triton.backends import compiler

print("triton file:", triton.__file__)
print("triton version:", getattr(triton, "__version__", "unknown"))
print("has compiler.Language:", hasattr(compiler, "Language"))
print("has tl.insert_slice:", hasattr(tl, "insert_slice"))
assert hasattr(compiler, "Language")
assert hasattr(tl, "insert_slice")
PY
```

## Install Or Refresh FLA-npu Ops

If you already built and installed FLA-npu successfully, you do not need to repeat this section. Keep the exact CANN path from your machine.

```bash
cd /home/canada_group_account/a00652497/bytedance/ascendc_triton_test/flash-linear-attention-npu
source /home/canada_group_account/CANN/9.0.0.0430/cann-9.0.0/set_env.sh
pip install -r requirements.txt
pip install pybind11

# Pick the SoC that matches the server: ascend910b, ascend910_93, or ascend950.
bash build.sh --soc=ascend910_93 --pkg --vendor_name=fla_npu
./build_out/fla-npu-*.run

cd torch_custom/fla_npu
bash build.sh
```

After installing the `.run` package, export the custom op API library path before running tests:

```bash
export LD_LIBRARY_PATH=/home/canada_group_account/CANN/9.0.0.0430/cann-9.0.0/opp/vendors/fla_npu_transformer/op_api/lib/:${LD_LIBRARY_PATH}
```

Do not run the precision script from inside `flash-linear-attention-npu/torch_custom/fla_npu`; that source directory can shadow the installed `fla_npu` wheel.

## Verify Runtime

From the test repo root:

```bash
cd /home/canada_group_account/a00652497/bytedance/ascendc_triton_test
export LD_LIBRARY_PATH=/home/canada_group_account/CANN/9.0.0.0430/cann-9.0.0/opp/vendors/fla_npu_transformer/op_api/lib/:${LD_LIBRARY_PATH}
python - <<'PY'
import torch
import torch_npu
import fla_npu

print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("npu", torch.npu.is_available())

ops = [
    "npu_recompute_w_u_fwd",
    "npu_chunk_gated_delta_rule_fwd_h",
    "npu_chunk_fwd_o",
    "npu_chunk_bwd_dv_local",
    "npu_chunk_gated_delta_rule_bwd_dhu",
    "npu_chunk_bwd_dqkwg",
    "npu_prepare_wy_repr_bwd_da",
    "npu_prepare_wy_repr_bwd_full",
]
for op in ops:
    print(op, hasattr(torch.ops.npu, op))
PY
```

All ops should print `True`.

## Run The Staged Precision Suite

Pull the current script first. If you see `No module named mindspeed_mm`, you are still running the old version.

```bash
cd /home/canada_group_account/a00652497/bytedance/ascendc_triton_test
git pull
export LD_LIBRARY_PATH=/home/canada_group_account/CANN/9.0.0.0430/cann-9.0.0/opp/vendors/fla_npu_transformer/op_api/lib/:${LD_LIBRARY_PATH}
bash run_precision_suite.sh
```

The suite writes logs and JSON under `./precision_results/`, then prints `./precision_results/summary.txt`. It runs fixed-length sanity first, then variable-length cases in this order:

The bundled fixed-length cases use `K=128,V=128`; smaller K/V dimensions such as 64 are intentionally not used because the FLA-npu AscendC kernels are validated around `K=128` and can fail host-side tiling before a precision comparison starts.

- `varlen_single_1024`: one packed segment, should behave like fixed length.
- `varlen_single_1121`: one segment with total length not divisible by `chunk_size`.
- `varlen_aligned_1024`: multiple sequences with lengths aligned to `chunk_size`.
- `varlen_unaligned_1121`: mixed non-aligned sequence lengths, total length not divisible by `chunk_size`.
- `varlen_unaligned_1152`: same mixed lengths plus one tail segment so total length is divisible by `chunk_size`.
- `varlen_short_mixed_1024`: very short and boundary-adjacent sequence lengths.
- `varlen_many_2048`: more segments with mixed chunk counts.

Interpretation:

- Fixed-length failures mean the full wrapper/reference/environment is not clean; do not debug varlen yet.
- Fixed passes but `varlen_single_1024` fails points to the varlen code path itself.
- Single-segment passes but aligned multi-segment fails points to per-sequence reset or `cu_seqlens` handling.
- `varlen_single_1121` failing points to packed total-length tail handling.
- `varlen_unaligned_1121` failing but `varlen_unaligned_1152` passing points to packed total length not being a `chunk_size` multiple.
- Aligned passes but unaligned/short cases fail points to partial chunk or `chunk_indices` handling.
- Forward output passes but gradients fail means the forward varlen metadata is probably OK; start drilling into backward ops.

Run a larger optional pass only after the default suite is understood:

```bash
RUN_LARGE=1 bash run_precision_suite.sh
```

## Locate The Bad Backward Branch

When the suite shows only `grad_k` failing for partial-tail varlen cases, run the operator-level diagnostic:

```bash
python diagnose_gradk_operator.py \
  --case varlen \
  --fla-repo ./flash-linear-attention-npu \
  --device 0 \
  --output-json ./precision_results/diagnose_varlen_1121.json
```

For the single-segment partial-tail repro:

```bash
python diagnose_gradk_operator.py \
  --case varlen \
  --cu-seqlens 0,1121 \
  --heads 8 \
  --key-dim 128 \
  --value-dim 128 \
  --chunk-size 64 \
  --fla-repo ./flash-linear-attention-npu \
  --device 0 \
  --tail-topk 16 \
  --output-json ./precision_results/diagnose_single_1121.json
```

For partial-tail failures, focus on these extra lines in the output:

- `packed_tail_grad_k`: metrics only for the final packed tail block, e.g. tokens `[1088,1121)` when `T=1121, chunk_size=64`.
- `non_tail_grad_k`: metrics for all valid tokens outside that final packed tail block.
- `top grad_k errors in packed final tail`: concrete `(packed_token, seq, token, head, dim)` coordinates and values for the largest tail errors.

The diagnostic also writes the same tail reports into JSON under each variant's `tail_reports`.

The diagnostic runs these variants:

- `ascendc`: real FLA-npu `flash_gated_delta_rule(...).backward()` wrapper path. This must reproduce `compare_gdn_precision.py`.
- `manual_ascendc`: hand-reconstructed internal AscendC op chain used only for debugging. It is not a deployable replacement.
- `triton_full`: complete local Triton GDN forward/backward. This is the Triton baseline; it does not mix AscendC intermediates.
- `triton_dqkwg`: only `npu_chunk_bwd_dqkwg` is replaced by the local Triton kernel.
- `triton_wy`: only `npu_prepare_wy_repr_bwd_da/full` is replaced by the local Triton WY backward.
- `triton_both`: replace `dqkwg` and WY backward, but keep `bwd_dhu` as AscendC.
- `triton_dhu`: experimental hybrid that replaces only `npu_chunk_gated_delta_rule_bwd_dhu`; it is not included in default `--variant all` because it can hit Triton-Ascend UB-limit compilation failures when isolated from the full Triton graph.
- `triton_dhu_dqkwg` / `triton_all`: experimental hybrids involving `bwd_dhu`; run them only with `--include-dhu-hybrid` or explicit `--variant`.

Read the result as follows:

- If `ascendc` fails but `manual_ascendc` passes, the diagnostic has not reproduced the real wrapper failure. Do not blame `dqkwg` or WY from that run; inspect wrapper/autograd parity first.
- If `ascendc` and `manual_ascendc` both fail, and `triton_dqkwg` passes, the likely bad operator is `npu_chunk_bwd_dqkwg`.
- If `ascendc` and `manual_ascendc` both fail, and `triton_wy` passes, the likely bad operator is `npu_prepare_wy_repr_bwd_da/full`.
- If only `triton_both` or `triton_all` passes, the error is split across multiple backward branches.
- If no Triton replacement passes, check earlier forward/backward intermediates (`h`, `v_new`, `A`) or a layout mismatch in the diagnostic.

The script also prints component deltas for `dk_from_dqkwg` and `dk_from_wy`, so the final verdict is not based only on one full-gradient pass/fail.

To avoid trusting an invalid hybrid replacement, run the gated bisect suite once:

```bash
bash run_gradk_bisect_suite.sh
```

It runs the same variants over three control cases (`fixed_1k_h8`, `varlen_single_1024`, `varlen_aligned_1024`) and the failing target (`target_single_1121`). The control cases are not new bug repros; they only prove a replacement does not break known-good fixed/aligned-varlen behavior.

After the controls have passed, skip them and run only the failing partial-tail target:

```bash
MODE=target bash run_gradk_bisect_suite.sh
```

To reduce compile/runtime further, run one candidate at a time:

```bash
MODE=target VARIANT=ascendc bash run_gradk_bisect_suite.sh
MODE=target VARIANT=triton_dqkwg bash run_gradk_bisect_suite.sh
MODE=target VARIANT=triton_wy bash run_gradk_bisect_suite.sh
MODE=target VARIANT=triton_both bash run_gradk_bisect_suite.sh
```

Only treat a replacement as usable if the full gated summary says:

- `controls_ok=True`: the replacement preserves known-good cases.
- `target_grad_k_ok=True`: the replacement fixes the failing partial-tail case.

The suite also prints `diagnostic gate`. If it says `wrapper fails but manual internal chain passes`, ignore `replacement validity`: the hand-reconstructed chain is missing some real wrapper behavior, so late-op replacement conclusions are not valid yet.

`triton_full` is the baseline for "pure Triton is good on this shape"; `triton_dqkwg`, `triton_wy`, and `triton_both` are candidate operator replacements. Do not use `triton_dhu` conclusions unless the explicit `bwd_dhu` hybrid first proves it can compile and pass the control cases.

To test the creative/MindSpeed-MM wrapper instead of the FLA-npu standalone example, run the same suite with `IMPL=creative` and point `CREATIVE_REPO` at that checkout:

```bash
IMPL=creative CREATIVE_REPO=/path/to/qwen3.5_omni_creative bash run_precision_suite.sh
```

The two implementations use the same GDN math structure, but they are not byte-for-byte identical: the default `fla` mode loads `flash-linear-attention-npu/examples/flash_gated_delta_rule.py`, while `creative` loads `mindspeed_mm/fsdp/models/qwen3_5/flash_gated_delta_rule.py` by file path.

## Run Individual Cases

Run one packed varlen case:

```bash
python compare_gdn_precision.py \
  --case varlen \
  --fla-repo ./flash-linear-attention-npu \
  --device 0 \
  --output-json ./gdn_varlen.json
```

If the FLA-npu repo is elsewhere, point `--fla-repo` to that checkout:

```bash
python compare_gdn_precision.py --case varlen --fla-repo /path/to/flash-linear-attention-npu --device 0
```

Run the quick fixed-length smoke case:

```bash
python compare_gdn_precision.py --case small --fla-repo ./flash-linear-attention-npu --device 0
```

Run all bundled cases:

```bash
python compare_gdn_precision.py --case all --fla-repo ./flash-linear-attention-npu --device 0 --output-json ./gdn_all.json
```

Run a custom packed varlen shape:

```bash
python compare_gdn_precision.py \
  --case varlen \
  --fla-repo ./flash-linear-attention-npu \
  --cu-seqlens 0,112,209,240,281,489,523,566,689,721,785,837,985,1071,1121 \
  --heads 8 \
  --key-dim 128 \
  --value-dim 128 \
  --chunk-size 64 \
  --device 0
```

## Output

The script prints JSON. Each compared tensor includes:

- `allclose`: result under `--atol` and `--rtol`
- `max_abs`: maximum absolute difference
- `mean_abs`: mean absolute difference
- `rms`: root mean square difference
- `max_rel`: maximum relative difference
- `mismatch_ratio`: fraction of elements outside tolerance

Exit codes:

- `0`: all compared tensors passed
- `1`: the run completed but at least one tensor failed tolerance
- `2`: environment/runtime error

Default tolerance is `atol=1e-2`, `rtol=1e-2`.

## Common Errors

- `No module named mindspeed_mm`: run `git pull`; the current script has no MindSpeed-MM import.
- `No module named triton` or Triton API errors such as missing `compiler.Language` / `tl.insert_slice`: clean mixed community Triton installs and reinstall `triton-ascend==3.2.0` as shown above.
- `fla_npu failed to import`: install `torch_custom/fla_npu`, export `LD_LIBRARY_PATH`, and run outside `torch_custom/fla_npu`.
- Missing `torch.ops.npu.*`: reinstall the FLA-npu `.run` package and rebuild the `fla_npu` wheel in the same Python environment.
