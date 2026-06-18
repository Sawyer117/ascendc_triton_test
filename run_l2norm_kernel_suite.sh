#!/usr/bin/env bash
set -u

PYTHON=${PYTHON:-python}
FLA_REPO=${FLA_REPO:-${FLA_NPU_REPO:-./flash-linear-attention-npu}}
DEVICE=${DEVICE:-0}
DTYPE=${DTYPE:-bf16}
OUT_DIR=${OUT_DIR:-./precision_results/l2norm_kernel}
LAYOUTS=${LAYOUTS:-BHTD}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${OUT_DIR}"

status=0

run_case() {
  local name=$1
  shift
  for layout in ${LAYOUTS}; do
    local tag="${name}_${layout}"
    local json="${OUT_DIR}/${tag}.json"
    local log="${OUT_DIR}/${tag}.log"
    local cmd=(
      "${PYTHON}" "${SCRIPT_DIR}/compare_l2norm_kernel_precision.py"
      --fla-repo "${FLA_REPO}"
      --device "${DEVICE}"
      --dtype "${DTYPE}"
      --layout "${layout}"
      --output-json "${json}"
      "$@"
    )
    echo "==> l2norm-kernel ${tag}"
    printf '    %q' "${cmd[@]}"
    echo
    if "${cmd[@]}" >"${log}" 2>&1; then
      echo "    PASS"
    else
      rc=$?
      status=1
      echo "    FAIL rc=${rc}; tail ${log}:"
      tail -n 40 "${log}" | sed 's/^/      /'
    fi
  done
}

echo "real l2norm kernel suite"
echo "  python:   ${PYTHON}"
echo "  fla_repo: ${FLA_REPO}"
echo "  device:   ${DEVICE}"
echo "  dtype:    ${DTYPE}"
echo "  layouts:  ${LAYOUTS}"
echo "  out_dir:  ${OUT_DIR}"
echo

run_case fixed_1k_h8 \
  --case small --seq-len 1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

run_case fixed_1121_h8 \
  --case small --seq-len 1121 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

run_case varlen_single_1024 \
  --case varlen --cu-seqlens 0,1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

run_case varlen_aligned_1024 \
  --case varlen --cu-seqlens 0,256,512,768,1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

run_case varlen_single_1121 \
  --case varlen --cu-seqlens 0,1121 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

run_case varlen_unaligned_1121 \
  --case varlen --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

"${PYTHON}" - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
rows = []
for path in sorted(out_dir.glob("*.json")):
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # pylint: disable=broad-except
        rows.append(f"{path.stem}\tjson_error={exc}")
        continue
    if payload.get("error"):
        rows.append(f"{path.stem}\terror={payload.get('error')}")
        continue
    comps = payload.get("comparisons", {})
    tails = payload.get("tail_reports", {})
    norm = comps.get("bwd_kernel_normalized_vs_autograd", {})
    orig = comps.get("bwd_kernel_original_vs_autograd", {})
    py_norm = comps.get("bwd_py_normalized_vs_autograd", {})
    py_orig = comps.get("bwd_py_original_vs_autograd", {})
    tail_norm = tails.get("bwd_kernel_normalized_vs_autograd", {})
    tail_orig = tails.get("bwd_kernel_original_vs_autograd", {})
    rows.append(
        "\t".join(
            [
                path.stem,
                f"passed={payload.get('passed')}",
                f"shape={payload.get('shape')}",
                f"layout={payload.get('layout')}",
                f"segment_tails={payload.get('segment_tails')}",
                f"kernel_norm allclose={norm.get('allclose')} max_abs={norm.get('max_abs')} rms={norm.get('rms')} mismatch={norm.get('mismatch_ratio')}",
                f"kernel_orig allclose={orig.get('allclose')} max_abs={orig.get('max_abs')} rms={orig.get('rms')} mismatch={orig.get('mismatch_ratio')}",
                f"py_norm allclose={py_norm.get('allclose')} max_abs={py_norm.get('max_abs')} rms={py_norm.get('rms')}",
                f"py_orig allclose={py_orig.get('allclose')} max_abs={py_orig.get('max_abs')} rms={py_orig.get('rms')}",
                f"tail_norm allclose={tail_norm.get('allclose')} max_abs={tail_norm.get('max_abs')} rms={tail_norm.get('rms')}",
                f"tail_orig allclose={tail_orig.get('allclose')} max_abs={tail_orig.get('max_abs')} rms={tail_orig.get('rms')}",
            ]
        )
    )

summary = out_dir / "summary.txt"
summary.write_text("\n".join(rows) + ("\n" if rows else ""))
print(f"summary: {summary}")
print(summary.read_text())
PY

exit "${status}"
