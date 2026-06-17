#!/usr/bin/env bash
set -u

PYTHON=${PYTHON:-python}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CREATIVE_REPO=${CREATIVE_REPO:-${SCRIPT_DIR}/creative_snapshot}
DEVICE=${DEVICE:-0}
DTYPE=${DTYPE:-bf16}
OUT_DIR=${OUT_DIR:-./creative_pair_results}
QK_L2NORM=${QK_L2NORM:-0}

mkdir -p "${OUT_DIR}"

l2norm_args=()
if [[ "${QK_L2NORM}" != "1" ]]; then
  l2norm_args+=(--no-qk-l2norm)
fi

run_case() {
  local name=$1
  shift
  local json="${OUT_DIR}/${name}.json"
  local log="${OUT_DIR}/${name}.log"
  local cmd=(
    "${PYTHON}" "${SCRIPT_DIR}/compare_creative_gdn_pair.py"
    --creative-repo "${CREATIVE_REPO}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
    --output-json "${json}"
    "${l2norm_args[@]}"
    "$@"
  )

  echo "==> ${name}"
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

echo "creative GDN pair suite"
echo "  python:        ${PYTHON}"
echo "  creative_repo: ${CREATIVE_REPO}"
echo "  device:        ${DEVICE}"
echo "  dtype:         ${DTYPE}"
echo "  qk_l2norm:    ${QK_L2NORM}"
echo "  out_dir:       ${OUT_DIR}"
echo

run_case fixed_1k_h8 \
  --case small --seq-len 1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

run_case varlen_single_1024 \
  --case varlen --cu-seqlens 0,1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

run_case varlen_aligned_1024 \
  --case varlen --cu-seqlens 0,256,512,768,1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

run_case varlen_single_1121 \
  --case varlen --cu-seqlens 0,1121 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64 --tail-topk 16

if [[ "${RUN_UNSAFE:-0}" == "1" ]]; then
  # These currently exercise a path that can raise an AICore out-of-range
  # exception on the installed custom op stack. Keep them out of the default
  # suite so normal precision runs do not poison the device context.
  run_case varlen_unaligned_1121 \
    --case varlen --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64 --tail-topk 16

  run_case varlen_unaligned_1152 \
    --case varlen --cu-seqlens 0,112,209,240,281,489,523,566,689,721,785,837,985,1071,1121,1152 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64 --tail-topk 16
fi

if [[ "${RUN_LARGE:-0}" == "1" ]]; then
  run_case fixed_medium \
    --case medium
  run_case varlen_many_2048 \
    --case varlen --cu-seqlens 0,16,33,97,128,193,257,384,513,777,1024,1536,2048 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64 --tail-topk 16
fi

"${PYTHON}" - "${OUT_DIR}" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(out_dir.glob("*.json")):
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # pylint: disable=broad-except
        rows.append((path.stem, "json_error", str(exc)))
        continue
    case = payload.get("cases", [{}])[0]
    comps = case.get("comparisons", {})
    failed = case.get("failed_tensors", [])
    worst_name = "-"
    worst = None
    for name, stats in comps.items():
        if worst is None or stats.get("max_abs", -1) > worst.get("max_abs", -1):
            worst_name, worst = name, stats
    tail = case.get("tail_reports", {}).get("packed_final_tail_grad_k")
    rows.append((path.stem, payload.get("passed"), failed, worst_name, worst, tail, case.get("mindspeed_triton_shim_used")))

summary_path = out_dir / "summary.txt"
with summary_path.open("w", encoding="utf-8") as f:
    for row in rows:
        if len(row) == 3:
            f.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")
            continue
        name, passed, failed, worst_name, worst, tail, shim = row
        worst_text = "-" if worst is None else f"{worst_name} max_abs={worst['max_abs']:.6g} rms={worst['rms']:.6g} mismatch={worst['mismatch_ratio']:.6g}"
        tail_text = "tail=-"
        if tail:
            stats = tail["stats"]
            tail_text = f"tail_grad_k allclose={stats['allclose']} max_abs={stats['max_abs']:.6g} rms={stats['rms']:.6g} mismatch={stats['mismatch_ratio']:.6g}"
        f.write(
            f"{name}\tpassed={passed}\tfailed={','.join(failed) or '-'}\t{worst_text}\t{tail_text}\tshim={shim}\n"
        )

print(f"summary: {summary_path}")
print(summary_path.read_text(), end="")
PY
