#!/usr/bin/env bash
set -u

PYTHON=${PYTHON:-python}
FLA_REPO=${FLA_REPO:-./flash-linear-attention-npu}
DEVICE=${DEVICE:-0}
DTYPE=${DTYPE:-bf16}
OUT_DIR=${OUT_DIR:-./precision_results/gradk_bisect}
MODE=${MODE:-controls_and_target}
VARIANT=${VARIANT:-all}
MANUAL_L2NORM_BWD_INPUT=${MANUAL_L2NORM_BWD_INPUT:-normalized}

mkdir -p "${OUT_DIR}"

echo "grad-k bisect suite"
echo "  python:   ${PYTHON}"
echo "  fla_repo: ${FLA_REPO}"
echo "  device:   ${DEVICE}"
echo "  dtype:    ${DTYPE}"
echo "  out_dir:  ${OUT_DIR}"
echo "  mode:     ${MODE}"
echo "  variant:  ${VARIANT}"
echo "  manual_l2norm_bwd_input: ${MANUAL_L2NORM_BWD_INPUT}"
echo

run_case() {
  local name=$1
  shift
  local json="${OUT_DIR}/${name}.json"
  local log="${OUT_DIR}/${name}.log"
  local cmd=(
    "${PYTHON}" diagnose_gradk_operator.py
    --fla-repo "${FLA_REPO}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
    --variant "${VARIANT}"
    --manual-l2norm-bwd-input "${MANUAL_L2NORM_BWD_INPUT}"
    --tail-topk 8
    --output-json "${json}"
    "$@"
  )

  echo "==> ${name}"
  printf '    %q' "${cmd[@]}"
  echo
  "${cmd[@]}" >"${log}" 2>&1
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "    FAIL rc=${rc}; tail ${log}:"
    tail -n 40 "${log}"
  else
    echo "    done"
  fi
}

case "${MODE}" in
  controls_and_target|all)
    run_case fixed_1k_h8 \
      --case small --seq-len 1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

    run_case varlen_single_1024 \
      --case varlen --cu-seqlens 0,1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

    run_case varlen_aligned_1024 \
      --case varlen --cu-seqlens 0,256,512,768,1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

    run_case target_single_1121 \
      --case varlen --cu-seqlens 0,1121 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64 --tail-topk 16
    ;;
  controls)
    run_case fixed_1k_h8 \
      --case small --seq-len 1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

    run_case varlen_single_1024 \
      --case varlen --cu-seqlens 0,1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64

    run_case varlen_aligned_1024 \
      --case varlen --cu-seqlens 0,256,512,768,1024 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64
    ;;
  target)
    run_case target_single_1121 \
      --case varlen --cu-seqlens 0,1121 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64 --tail-topk 16
    ;;
  *)
    echo "unknown MODE=${MODE}; expected controls_and_target, controls, or target" >&2
    exit 2
    ;;
esac

"${PYTHON}" - "${OUT_DIR}" "${VARIANT}" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
variant_env = sys.argv[2]
case_files = {
    "fixed_1k_h8": out_dir / "fixed_1k_h8.json",
    "varlen_single_1024": out_dir / "varlen_single_1024.json",
    "varlen_aligned_1024": out_dir / "varlen_aligned_1024.json",
    "target_single_1121": out_dir / "target_single_1121.json",
}
cases = {}
for name, path in case_files.items():
    if path.exists():
        cases[name] = json.loads(path.read_text())

if variant_env == "all":
    variants = ["ascendc", "manual_ascendc", "triton_full", "triton_dvlocal", "triton_dqkwg", "triton_wy", "triton_both"]
else:
    variants = [variant_env]
controls = ["fixed_1k_h8", "varlen_single_1024", "varlen_aligned_1024"]
target = "target_single_1121"

def variant_result(case_name, variant):
    return cases.get(case_name, {}).get("variants", {}).get(variant)

def grad_k_stats(result):
    if not result or "comparisons" not in result:
        return None
    return result["comparisons"].get("grad_k")

def tail_stats(result):
    if not result:
        return None
    return result.get("tail_reports", {}).get("packed_final_tail", {}).get("stats")

def ok(result):
    if not result or result.get("error"):
        return False
    stats = grad_k_stats(result)
    return bool(stats and stats.get("allclose"))

print()
print(f"summary: {out_dir}")
header = f"{'case':22s} {'variant':14s} {'grad_k':8s} {'max_abs':>10s} {'tail':8s} {'tail_max':>10s} error"
print(header)
for case_name in case_files:
    for variant in variants:
        result = variant_result(case_name, variant)
        stats = grad_k_stats(result)
        tail = tail_stats(result)
        if result is None:
            print(f"{case_name:22s} {variant:14s} missing")
            continue
        if result.get("error"):
            first = result["error"].splitlines()[0] if result.get("error") else "error"
            print(f"{case_name:22s} {variant:14s} error    {'-':>10s} {'-':8s} {'-':>10s} {first}")
            continue
        print(
            f"{case_name:22s} {variant:14s} "
            f"{str(stats['allclose']):8s} {stats['max_abs']:10.6g} "
            f"{str(tail['allclose']) if tail else '-':8s} "
            f"{tail['max_abs'] if tail else 0:10.6g} -"
        )

print()
print("diagnostic gate")
asc_target = variant_result(target, "ascendc")
manual_target = variant_result(target, "manual_ascendc")
if asc_target is not None or manual_target is not None:
    asc_ok = ok(asc_target)
    manual_ok = ok(manual_target)
    print(f"target_wrapper_grad_k_ok={asc_ok} target_manual_chain_grad_k_ok={manual_ok}")
    if not asc_ok and manual_ok:
        print("wrapper fails but manual internal chain passes; ignore replacement validity until wrapper/manual parity is resolved")

print()
print("replacement validity")
for variant in variants:
    if variant in ("ascendc", "manual_ascendc"):
        continue
    if not all(case in cases for case in controls):
        print(
            f"{variant:14s} "
            "controls_ok=not_checked "
            f"target_grad_k_ok={ok(variant_result(target, variant))}"
        )
        continue
    control_ok = all(ok(variant_result(case, variant)) for case in controls)
    target_ok = ok(variant_result(target, variant))
    print(
        f"{variant:14s} "
        f"controls_ok={control_ok} "
        f"target_grad_k_ok={target_ok}"
    )
PY
