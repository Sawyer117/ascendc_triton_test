# AscendC GDN Precision Test, No MindSpeed-MM

This repo contains a standalone precision script for the FLA-npu Qwen3.5 GDN path.
It does **not** import `mindspeed_mm`.

Current comparison:

- AscendC path: `flash-linear-attention-npu/examples/flash_gated_delta_rule.py`
- Baseline: pure PyTorch reference embedded in `compare_gdn_precision.py`
- Checked tensors: forward output plus gradients of `q`, `k`, `v`, `beta`, and `g`
- Supported cases: fixed length and packed varlen via `cu_seqlens`

The AscendC wrapper still imports helper kernels from the local FLA-npu repo, so you still need the FLA-npu Python dependencies such as `triton-ascend`. You do not need MindSpeed-MM.

## Expected Environment

Run on the Ascend server with the same Python environment that has working `torch`, `torch_npu`, and `fla_npu`.

Known-good family from the FLA-npu notes:

- CANN 8.5+ or 9.x
- Ascend PyTorch release `v26.1.0-beta.1`
- `torch==2.7.1`
- `torch_npu==2.7.1.post5`, matching your Python ABI and CPU arch
- `triton-ascend`, `pybind11`, and `flash-linear-attention-npu/requirements.txt`

If `torch_custom/fla_npu/build.sh` reports `No module named torchnpugen`, your `torch_npu` wheel is missing or from the wrong release family. The matching `torch_npu` wheel should provide `torchnpugen`.

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

## Run Varlen Precision

Pull the current script first. If you see `No module named mindspeed_mm`, you are still running the old version.

```bash
cd /home/canada_group_account/a00652497/bytedance/ascendc_triton_test
git pull
export LD_LIBRARY_PATH=/home/canada_group_account/CANN/9.0.0.0430/cann-9.0.0/opp/vendors/fla_npu_transformer/op_api/lib/:${LD_LIBRARY_PATH}
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
- `No module named triton`: install `triton-ascend`; avoid mixing it with community `triton` for older Ascend releases.
- `fla_npu failed to import`: install `torch_custom/fla_npu`, export `LD_LIBRARY_PATH`, and run outside `torch_custom/fla_npu`.
- Missing `torch.ops.npu.*`: reinstall the FLA-npu `.run` package and rebuild the `fla_npu` wheel in the same Python environment.
