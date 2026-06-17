# AscendC vs Triton GDN Precision Test

This repository contains a standalone precision comparison script for the
Qwen3.5 Gated DeltaNet (GDN) operator paths in MindSpeed-MM:

- Triton path: `mindspeed_mm.fsdp.models.qwen3_5.chunk_gated_delta_rule`
- AscendC path: `mindspeed_mm.fsdp.models.qwen3_5.flash_gated_delta_rule`

The script runs both implementations on identical random inputs and compares
the forward output plus gradients of `q`, `k`, `v`, `beta`, and `g`.

## 1. Expected Environment

Run this on an Ascend NPU host. The script is not useful on CPU-only machines.

Recommended baseline from the MindSpeed-MM Qwen3.5 notes:

- Python 3.10
- CANN 8.5.x or newer
- PyTorch and torch_npu matched to your PTA release
- MindSpeed-MM installed or available in `PYTHONPATH`
- Triton-on-NPU support for the original GDN path
- FLA-npu custom operators for the AscendC GDN path

The AscendC GDN adaptation notes in MindSpeed-MM mention a strict PTA
requirement and recommend the `26.1.0.beta` PyTorch Adapter for this path.

## 2. Install MindSpeed-MM

Clone and install MindSpeed-MM on the server. Example:

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM

# Adjust the MindSpeed commit ID to the one required by your branch.
bash scripts/install.sh --msid eb10b92
bash examples/qwen3_5/install_extensions.sh
```

If you already have a working MindSpeed-MM environment for Qwen3.5, keep using
that environment. The comparison script only needs to import the installed
`mindspeed_mm` package.

## 3. Install AscendC GDN Custom Ops

The AscendC path depends on FLA-npu custom operators. Build them from the FLA
NPU source repository on the target machine:

```bash
git clone https://github.com/flashserve/flash-linear-attention-npu.git
cd flash-linear-attention-npu

# Change this path to your actual CANN toolkit path.
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Select the correct SoC for your machine:
#   ascend910b, ascend910_93, or ascend950
bash build.sh --soc=ascend910_93 --pkg --vendor_name=fla_npu

# Install the generated custom-op run package.
./build_out/fla-npu-*.run

# Build and install the torch adapter wheel.
cd torch_custom/fla_npu
bash build.sh
```

Optional single-op smoke test from the FLA-npu repo:

```bash
cd torch_custom/fla_npu/test
bash test.sh
```

## 4. Verify Imports

From the same Python environment:

```bash
python - <<'PY'
import torch
import torch_npu
import fla_npu

print("torch:", torch.__version__)
print("npu available:", torch.npu.is_available())
print("fla_npu imported")
PY
```

If `fla_npu` imports but `torch.ops.npu.npu_chunk_*` is missing at runtime,
recheck that the custom-op run package and torch adapter wheel were installed
after sourcing the correct CANN environment.

## 5. Run Precision Comparison

Clone this repo on the server:

```bash
git clone https://github.com/Sawyer117/ascendc_triton_test.git
cd ascendc_triton_test
```

If MindSpeed-MM is installed editable, this may be enough:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python compare_gdn_precision.py --case small
```

If MindSpeed-MM is only available as source, set `PYTHONPATH`:

```bash
export MINDSPEED_MM=/path/to/MindSpeed-MM
source /usr/local/Ascend/ascend-toolkit/set_env.sh
PYTHONPATH=${MINDSPEED_MM}:${PYTHONPATH} \
  python compare_gdn_precision.py --case all --output-json /tmp/gdn_precision.json
```

Run a custom shape:

```bash
PYTHONPATH=${MINDSPEED_MM}:${PYTHONPATH} \
  python compare_gdn_precision.py \
  --batch 1 \
  --seq-len 4608 \
  --heads 32 \
  --key-dim 128 \
  --value-dim 128 \
  --chunk-size 64 \
  --atol 1e-2 \
  --rtol 1e-2
```

Run a packed variable-length case:

```bash
PYTHONPATH=${MINDSPEED_MM}:${PYTHONPATH} \
  python compare_gdn_precision.py \
  --batch 1 \
  --cu-seqlens 0,112,209,240,281,489,523,566,689,721,785,837,985,1071,1121 \
  --heads 8 \
  --key-dim 128 \
  --value-dim 128 \
  --chunk-size 64
```

## 6. Output Interpretation

The script prints JSON. Each compared tensor includes:

- `allclose`: result under the selected `--atol` and `--rtol`
- `max_abs`: maximum absolute difference
- `mean_abs`: mean absolute difference
- `rms`: root mean square difference
- `max_rel`: maximum relative difference
- `mismatch_ratio`: fraction of elements outside tolerance

Exit codes:

- `0`: all compared tensors passed `allclose`
- `1`: script ran, but at least one tensor failed tolerance
- `2`: environment or runtime error, such as missing `torch_npu`, `fla_npu`, NPU, or MindSpeed-MM

The default tolerance is `atol=1e-2`, `rtol=1e-2`, matching the existing
MindSpeed-MM Qwen3.5 GDN unit-test tolerance style.

## 7. Common Problems

- `No module named mindspeed_mm`: install MindSpeed-MM or set `PYTHONPATH` to
  the MindSpeed-MM source root.
- `torch_npu is required`: activate the Ascend PyTorch environment.
- `fla_npu is required`: build and install FLA-npu custom ops first.
- `NPU is not available`: run on an Ascend host and source the CANN environment.
- `torch.ops.npu.npu_chunk_*` missing: reinstall the FLA-npu run package and
  torch adapter wheel under the same CANN/PTA environment.
