#!/usr/bin/env bash
set -u

PYTHON=${PYTHON:-python}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FLA_REPO=${FLA_REPO:-./flash-linear-attention-npu}
DEVICE=${DEVICE:-0}
DTYPE=${DTYPE:-bf16}
OUT_DIR=${OUT_DIR:-./precision_results/gradk_trace}
MODE=${MODE:-target}

mkdir -p "${OUT_DIR}"

echo "grad-k component trace suite"
echo "  python:   ${PYTHON}"
echo "  fla_repo: ${FLA_REPO}"
echo "  device:   ${DEVICE}"
echo "  dtype:    ${DTYPE}"
echo "  out_dir:  ${OUT_DIR}"
echo "  mode:     ${MODE}"
echo

run_trace() {
  local name=$1
  shift
  local json="${OUT_DIR}/${name}.json"
  local log="${OUT_DIR}/${name}.log"
  local cmd=(
    "${PYTHON}" "${SCRIPT_DIR}/trace_gradk_components.py"
    --fla-repo "${FLA_REPO}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
    --output-json "${json}"
    "$@"
  )

  echo "==> ${name}"
  printf '    %q' "${cmd[@]}"
  echo
  "${cmd[@]}" >"${log}" 2>&1
  local rc=$?
  if [[ ${rc} -eq 0 ]]; then
    echo "    PASS all components"
  elif [[ ${rc} -eq 1 ]]; then
    echo "    TRACE found divergence; tail ${log}:"
    tail -n 60 "${log}"
  else
    echo "    ERROR rc=${rc}; tail ${log}:"
    tail -n 80 "${log}"
  fi
}

case "${MODE}" in
  target)
    run_trace target_single_1121 \
      --case varlen --cu-seqlens 0,1121 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64 --tail-topk 16
    ;;
  controls_and_target|all)
    run_trace fixed_1k_h8 \
      --case small --seq-len 1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64
    run_trace varlen_single_1024 \
      --case varlen --cu-seqlens 0,1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64
    run_trace varlen_aligned_1024 \
      --case varlen --cu-seqlens 0,256,512,768,1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64
    run_trace target_single_1121 \
      --case varlen --cu-seqlens 0,1121 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64 --tail-topk 16
    ;;
  *)
    echo "unknown MODE=${MODE}; expected target or controls_and_target" >&2
    exit 2
    ;;
esac

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
        rows.append(f"{path.stem}\tjson_error={exc}")
        continue
    first = payload.get("first_failed_component")
    rows.append(f"{path.stem}\tfirst_failed_component={first}")
    comps = payload.get("components", {})
    for name, item in comps.items():
        stats = item.get("all", {})
        tail = item.get("tail")
        if tail:
            rows.append(
                f"{path.stem}\t{name}\tallclose={stats.get('allclose')}"
                f"\tmax_abs={stats.get('max_abs')}\trms={stats.get('rms')}"
                f"\ttail_allclose={tail.get('allclose')}\ttail_max_abs={tail.get('max_abs')}\ttail_rms={tail.get('rms')}"
            )
        else:
            rows.append(
                f"{path.stem}\t{name}\tallclose={stats.get('allclose')}"
                f"\tmax_abs={stats.get('max_abs')}\trms={stats.get('rms')}"
            )
summary_path = out_dir / "summary.txt"
summary_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"summary: {summary_path}")
print(summary_path.read_text(), end="")
PY
