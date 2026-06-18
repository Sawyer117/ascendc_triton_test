#!/usr/bin/env bash
set -u

PYTHON=${PYTHON:-python}
FLA_REPO=${FLA_REPO:-${FLA_NPU_REPO:-./flash-linear-attention-npu}}
DEVICE=${DEVICE:-0}
DTYPE=${DTYPE:-bf16}
OUT_DIR=${OUT_DIR:-./precision_results/l2norm_context}
MODE=${MODE:-target}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${OUT_DIR}"
status=0

run_case() {
  local name=$1
  shift
  local json="${OUT_DIR}/${name}.json"
  local log="${OUT_DIR}/${name}.log"
  local cmd=(
    "${PYTHON}" "${SCRIPT_DIR}/compare_l2norm_context_precision.py"
    --fla-repo "${FLA_REPO}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
    --output-json "${json}"
    "$@"
  )
  echo "==> l2norm-context ${name}"
  printf '    %q' "${cmd[@]}"
  echo
  if "${cmd[@]}" >"${log}" 2>&1; then
    echo "    PASS"
  else
    rc=$?
    status=1
    echo "    FAIL rc=${rc}; tail ${log}:"
    tail -n 60 "${log}" | sed 's/^/      /'
  fi
}

echo "l2norm GDN-context suite"
echo "  python:   ${PYTHON}"
echo "  fla_repo: ${FLA_REPO}"
echo "  device:   ${DEVICE}"
echo "  dtype:    ${DTYPE}"
echo "  out_dir:  ${OUT_DIR}"
echo "  mode:     ${MODE}"
echo

case "${MODE}" in
  target)
    run_case target_single_1121 \
      --case varlen --cu-seqlens 0,1121 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64
    ;;
  controls_and_target)
    run_case fixed_1k_h8 \
      --case small --seq-len 1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64
    run_case varlen_single_1024 \
      --case varlen --cu-seqlens 0,1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64
    run_case varlen_aligned_1024 \
      --case varlen --cu-seqlens 0,256,512,768,1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64
    run_case target_single_1121 \
      --case varlen --cu-seqlens 0,1121 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64
    ;;
  *)
    echo "unknown MODE=${MODE}; expected target or controls_and_target" >&2
    exit 2
    ;;
esac

"${PYTHON}" - "${OUT_DIR}" <<'INNERPY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
rows = []
for path in sorted(out_dir.glob('*.json')):
    payload = json.loads(path.read_text())
    if payload.get('error'):
        rows.append(f"{path.stem}\terror={payload.get('error')}")
        continue
    comps = payload.get('comparisons', {})
    tails = payload.get('tail_reports', {})
    dk_stats = (payload.get('tensor_value_stats') or {}).get('dk_core') or {}
    dk_tail_stats = (payload.get('tensor_tail_value_stats') or {}).get('dk_core') or {}
    topk = ((payload.get('topk_errors') or {}).get('k_kernel_vs_py_norm') or [{}])[0]
    tail_topk = ((payload.get('topk_errors') or {}).get('tail_k_kernel_vs_py_norm') or [{}])[0]
    fields = [
        path.stem,
        f"passed={payload.get('passed')}",
        f"shape={payload.get('shape_bhtd')}",
        f"segment_tails={payload.get('segment_tails')}",
        f"dk_core min={dk_stats.get('min')} max={dk_stats.get('max')} rms={dk_stats.get('rms')} zero={dk_stats.get('zero_ratio')}",
        f"tail_dk_core min={dk_tail_stats.get('min')} max={dk_tail_stats.get('max')} rms={dk_tail_stats.get('rms')} zero={dk_tail_stats.get('zero_ratio')}",
        f"top1 abs={topk.get('abs_diff')} idx=({topk.get('batch')},{topk.get('head')},{topk.get('packed_token')},{topk.get('dim')}) seq={topk.get('seq')} tok={topk.get('token_in_seq')} tail={topk.get('in_tail')} kernel={topk.get('kernel')} py={topk.get('py')} dk={topk.get('dk_core')} x={topk.get('x_norm')} rstd={topk.get('rstd')}",
        f"tail_top1 abs={tail_topk.get('abs_diff')} idx=({tail_topk.get('batch')},{tail_topk.get('head')},{tail_topk.get('packed_token')},{tail_topk.get('dim')}) seq={tail_topk.get('seq')} tok={tail_topk.get('token_in_seq')} kernel={tail_topk.get('kernel')} py={tail_topk.get('py')} dk={tail_topk.get('dk_core')} x={tail_topk.get('x_norm')} rstd={tail_topk.get('rstd')}",
    ]
    for key in (
        'q_kernel_vs_py_norm',
        'k_kernel_vs_py_norm',
        'k_kernel_dy_contig_vs_py_norm',
        'k_kernel_dy_clone_vs_py_norm',
        'k_kernel_all_clone_vs_py_norm',
        'q_kernel_vs_ref',
        'k_kernel_vs_ref',
        'q_py_norm_vs_ref',
        'k_py_norm_vs_ref',
    ):
        item = comps.get(key, {})
        fields.append(
            f"{key} allclose={item.get('allclose')} max_abs={item.get('max_abs')} rms={item.get('rms')} mismatch={item.get('mismatch_ratio')}"
        )
    for key in ('q_kernel_vs_ref', 'k_kernel_vs_ref'):
        item = tails.get(key, {})
        fields.append(f"tail_{key} allclose={item.get('allclose')} max_abs={item.get('max_abs')} rms={item.get('rms')}")
    rows.append('\t'.join(fields))

summary = out_dir / 'summary.txt'
summary.write_text('\n'.join(rows) + ('\n' if rows else ''))
print(f"summary: {summary}")
print(summary.read_text())
INNERPY

exit "${status}"
