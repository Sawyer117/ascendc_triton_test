#!/usr/bin/env bash
set -u

PYTHON=${PYTHON:-python}
FLA_REPO=${FLA_REPO:-./flash-linear-attention-npu}
DEVICE=${DEVICE:-0}
DTYPE=${DTYPE:-bf16}
OUT_DIR=${OUT_DIR:-./precision_results}
SUMMARY_TSV="${OUT_DIR}/runs.tsv"

mkdir -p "${OUT_DIR}"
: > "${SUMMARY_TSV}"
printf 'case\trc\tjson\tlog\n' >> "${SUMMARY_TSV}"

echo "precision suite"
echo "  python:   ${PYTHON}"
echo "  fla_repo: ${FLA_REPO}"
echo "  device:   ${DEVICE}"
echo "  dtype:    ${DTYPE}"
echo "  out_dir:  ${OUT_DIR}"
echo ""

run_case() {
  local name="$1"
  shift
  local json="${OUT_DIR}/${name}.json"
  local log="${OUT_DIR}/${name}.log"

  echo "==> ${name}"
  echo "    ${PYTHON} compare_gdn_precision.py --fla-repo ${FLA_REPO} --device ${DEVICE} --dtype ${DTYPE} --output-json ${json} $*"
  "${PYTHON}" compare_gdn_precision.py \
    --fla-repo "${FLA_REPO}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --output-json "${json}" \
    "$@" > "${log}" 2>&1
  local rc=$?
  printf '%s\t%s\t%s\t%s\n' "${name}" "${rc}" "${json}" "${log}" >> "${SUMMARY_TSV}"

  if [[ "${rc}" == "0" ]]; then
    echo "    PASS"
  else
    echo "    FAIL rc=${rc}"
    if [[ -s "${json}" ]]; then
      "${PYTHON}" - "${json}" <<'PY_FAIL'
import json
import sys

path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
case = payload["cases"][0]
print("    failed tensors:")
for name, stats in case["comparisons"].items():
    if not stats["allclose"]:
        print(
            f"      {name}: max_abs={stats['max_abs']:.6g} "
            f"rms={stats['rms']:.6g} mismatch={stats['mismatch_ratio']:.6g} "
            f"max_rel={stats['max_rel']:.6g}"
        )
PY_FAIL
    fi
    echo "    tail ${log}:"
    tail -n 30 "${log}"
  fi
  echo ""
}

# Fixed-length sanity. These establish that the full forward/backward chain is sane
# before investigating packed variable-length metadata and per-sequence resets.
run_case fixed_small --case small
run_case fixed_medium --case medium
run_case fixed_1k_h8 --case small --seq-len 1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64
run_case fixed_4k_h8 --case small --seq-len 4096 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

# Variable-length coverage. The order is intentional:
# 1) a single segment should behave like fixed length;
# 2) aligned segments avoid partial chunks;
# 3) unaligned/mixed segments stress partial chunks and chunk metadata.
run_case varlen_single_1024 --case varlen --cu-seqlens 0,1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64
run_case varlen_single_1121 --case varlen --cu-seqlens 0,1121 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64
run_case varlen_aligned_1024 --case varlen --cu-seqlens 0,256,512,768,1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64
run_case varlen_unaligned_1121 --case varlen
run_case varlen_unaligned_1152 --case varlen --cu-seqlens 0,112,209,240,281,489,523,566,689,721,785,837,985,1071,1121,1152 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64
run_case varlen_short_mixed_1024 --case varlen --cu-seqlens 0,1,17,63,64,65,127,128,129,257,511,1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64
run_case varlen_many_2048 --case varlen --cu-seqlens 0,16,33,97,128,193,257,384,513,777,1024,1536,2048 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

if [[ "${RUN_LARGE:-0}" == "1" ]]; then
  run_case fixed_8k_h16 --case small --seq-len 8192 --heads 16 --key-dim 128 --value-dim 128 --chunk-size 64
  run_case varlen_large_8192 --case varlen --cu-seqlens 0,16,1024,2048,4096,6144,8192 --heads 16 --key-dim 128 --value-dim 128 --chunk-size 64
fi

"${PYTHON}" - "${OUT_DIR}" > "${OUT_DIR}/summary.txt" <<'PY_SUMMARY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
for path in sorted(out_dir.glob("*.json")):
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"{path.stem}\tjson_error={exc}")
        continue
    case = payload["cases"][0]
    comps = case["comparisons"]
    worst_name, worst = max(comps.items(), key=lambda item: item[1]["max_abs"])
    failed = [name for name, stats in comps.items() if not stats["allclose"]]
    print(
        f"{path.stem}\tpassed={case['passed']}\t"
        f"worst={worst_name}\tmax_abs={worst['max_abs']:.6g}\t"
        f"rms={worst['rms']:.6g}\tmismatch={worst['mismatch_ratio']:.6g}\t"
        f"failed={','.join(failed) if failed else '-'}"
    )
PY_SUMMARY

echo "summary: ${OUT_DIR}/summary.txt"
cat "${OUT_DIR}/summary.txt"
