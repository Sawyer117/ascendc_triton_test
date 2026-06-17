#!/usr/bin/env bash
set -u

PYTHON=${PYTHON:-python}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CREATIVE_REPO=${CREATIVE_REPO:-${SCRIPT_DIR}/creative_snapshot}
DEVICE=${DEVICE:-0}
DTYPE=${DTYPE:-bf16}
OUT_DIR=${OUT_DIR:-./l2norm_mode_results}

mkdir -p "${OUT_DIR}"
find "${OUT_DIR}" -maxdepth 1 -type f \( -name '*.json' -o -name '*.log' -o -name 'summary.txt' \) -delete

run_gdn() {
  local name=$1
  local mode=$2
  shift 2
  local json="${OUT_DIR}/${name}_${mode}.json"
  local log="${OUT_DIR}/${name}_${mode}.log"
  local cmd=(
    "${PYTHON}" "${SCRIPT_DIR}/compare_creative_gdn_pair.py"
    --creative-repo "${CREATIVE_REPO}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
    --mixed-l2norm-mode "${mode}"
    --output-json "${json}"
    "$@"
  )

  echo "==> gdn ${name} ${mode}"
  printf '    %q' "${cmd[@]}"
  echo
  "${cmd[@]}" >"${log}" 2>&1
  local rc=$?
  if [[ ${rc} -eq 0 ]]; then
    echo "    PASS"
  elif [[ ${rc} -eq 1 ]]; then
    echo "    FAIL precision; tail ${log}:"
    tail -n 40 "${log}"
  else
    echo "    ERROR rc=${rc}; tail ${log}:"
    tail -n 60 "${log}"
  fi
}

run_l2norm() {
  local name=$1
  shift
  local json="${OUT_DIR}/l2norm_${name}.json"
  local log="${OUT_DIR}/l2norm_${name}.log"
  local cmd=(
    "${PYTHON}" "${SCRIPT_DIR}/compare_l2norm_precision.py"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
    --output-json "${json}"
    "$@"
  )

  echo "==> l2norm ${name} self-contained"
  printf '    %q' "${cmd[@]}"
  echo
  "${cmd[@]}" >"${log}" 2>&1
  local rc=$?
  if [[ ${rc} -eq 0 ]]; then
    echo "    PASS"
  elif [[ ${rc} -eq 3 ]]; then
    echo "    SKIP; tail ${log}:"
    tail -n 20 "${log}"
  elif [[ ${rc} -eq 1 ]]; then
    echo "    FAIL precision; tail ${log}:"
    tail -n 40 "${log}"
  else
    echo "    ERROR rc=${rc}; tail ${log}:"
    tail -n 60 "${log}"
  fi
}

run_gdn_pair() {
  local name=$1
  shift
  run_gdn "${name}" outer "$@"
  run_gdn "${name}" kernel "$@"
}

echo "l2norm mode suite"
echo "  python:        ${PYTHON}"
echo "  creative_repo: ${CREATIVE_REPO}"
echo "  device:        ${DEVICE}"
echo "  dtype:         ${DTYPE}"
echo "  out_dir:       ${OUT_DIR}"
echo

run_gdn_pair fixed_1k_h8 \
  --case small --seq-len 1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

run_gdn_pair varlen_single_1024 \
  --case varlen --cu-seqlens 0,1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

run_gdn_pair varlen_aligned_1024 \
  --case varlen --cu-seqlens 0,256,512,768,1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

run_gdn_pair varlen_single_1121 \
  --case varlen --cu-seqlens 0,1121 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64 --tail-topk 16

run_l2norm fixed_1k_h8 \
  --case small --seq-len 1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

run_l2norm varlen_single_1024 \
  --case varlen --cu-seqlens 0,1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

run_l2norm varlen_aligned_1024 \
  --case varlen --cu-seqlens 0,256,512,768,1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

run_l2norm varlen_single_1121 \
  --case varlen --cu-seqlens 0,1121 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

"${PYTHON}" - "${OUT_DIR}" <<'PYSUM'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(out_dir.glob("*.json")):
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # pylint: disable=broad-except
        rows.append(f"{path.stem}\tjson_error={exc}")
        continue
    if payload.get("skipped"):
        rows.append(f"{path.stem}\tskipped=True\treason={payload.get('skip_reason')}")
        continue
    if payload.get("comparison") == "creative_ascendc_mixed_vs_creative_pure_triton":
        case = payload.get("cases", [{}])[0]
        comps = case.get("comparisons", {})
        failed = case.get("failed_tensors", [])
        grad_q = comps.get("grad_q", {})
        grad_k = comps.get("grad_k", {})
        rows.append(
            f"{path.stem}\ttype=gdn\tmode={case.get('mixed_l2norm_mode')}\tpassed={payload.get('passed')}"
            f"\tfailed={','.join(failed) or '-'}"
            f"\tgrad_q max_abs={grad_q.get('max_abs', '-')} rms={grad_q.get('rms', '-')} mismatch={grad_q.get('mismatch_ratio', '-')}"
            f"\tgrad_k max_abs={grad_k.get('max_abs', '-')} rms={grad_k.get('rms', '-')} mismatch={grad_k.get('mismatch_ratio', '-')}"
        )
    else:
        comps = payload.get("comparisons", {})
        contracts = payload.get("contracts", {})
        out = comps.get("output", {})
        bwd_orig = comps.get("bwd_original_input_contract", {})
        bwd_norm = comps.get("bwd_normalized_output_contract", {})
        flash_orig = comps.get("flash_old_call_if_bwd_expects_original", {})
        flash_norm = comps.get("flash_old_call_if_bwd_expects_normalized", {})
        rows.append(
            f"{path.stem}\ttype=l2norm\tsource={payload.get('source')}\tpassed={payload.get('passed')}"
            f"\toriginal_contract_ok={contracts.get('original_input_contract_ok')}"
            f"\tnormalized_contract_ok={contracts.get('normalized_output_contract_ok')}"
            f"\told_flash_if_original_ok={contracts.get('old_flash_call_ok_if_bwd_expects_original')}"
            f"\told_flash_if_normalized_ok={contracts.get('old_flash_call_ok_if_bwd_expects_normalized')}"
            f"\toutput max_abs={out.get('max_abs', '-')} rms={out.get('rms', '-')}"
            f"\tbwd_orig max_abs={bwd_orig.get('max_abs', '-')} rms={bwd_orig.get('rms', '-')} mismatch={bwd_orig.get('mismatch_ratio', '-')}"
            f"\tbwd_norm max_abs={bwd_norm.get('max_abs', '-')} rms={bwd_norm.get('rms', '-')} mismatch={bwd_norm.get('mismatch_ratio', '-')}"
            f"\tflash_if_orig max_abs={flash_orig.get('max_abs', '-')} rms={flash_orig.get('rms', '-')} mismatch={flash_orig.get('mismatch_ratio', '-')}"
            f"\tflash_if_norm max_abs={flash_norm.get('max_abs', '-')} rms={flash_norm.get('rms', '-')} mismatch={flash_norm.get('mismatch_ratio', '-')}"
        )
summary_path = out_dir / "summary.txt"
summary_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"summary: {summary_path}")
print(summary_path.read_text(), end="")
PYSUM
