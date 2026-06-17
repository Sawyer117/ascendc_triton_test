#!/usr/bin/env bash
set -u

PYTHON=${PYTHON:-python}
FLA_REPO=${FLA_REPO:-./flash-linear-attention-npu}
DEVICE=${DEVICE:-0}
DTYPE=${DTYPE:-bf16}
OUT_DIR=${OUT_DIR:-./precision_results/gradk_bisect}
MODE=${MODE:-controls_and_target}
VARIANT=${VARIANT:-all}
VARIANT_LIST=${VARIANT_LIST:-}
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
if [[ -n "${VARIANT_LIST}" ]]; then
  echo "  variant_list: ${VARIANT_LIST}"
fi
echo "  manual_l2norm_bwd_input: ${MANUAL_L2NORM_BWD_INPUT}"
echo

run_case_with_variant() {
  local name=$1
  local variant=$2
  shift 2
  local json="${OUT_DIR}/${name}.json"
  local log="${OUT_DIR}/${name}.log"
  local cmd=(
    "${PYTHON}" diagnose_gradk_operator.py
    --fla-repo "${FLA_REPO}"
    --device "${DEVICE}"
    --dtype "${DTYPE}"
    --variant "${variant}"
    --manual-l2norm-bwd-input "${MANUAL_L2NORM_BWD_INPUT}"
    --tail-topk 8
    --output-json "${json}"
    "$@"
  )

  echo "==> ${name} (${variant})"
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

run_case() {
  local name=$1
  shift
  run_case_with_variant "${name}" "${VARIANT}" "$@"
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
  target_isolated)
    if [[ -n "${VARIANT_LIST}" ]]; then
      read -r -a variants <<<"${VARIANT_LIST}"
    else
      variants=(ascendc ascendc_saved_x manual_ascendc triton_full triton_dqkwg triton_wy triton_both triton_dvlocal)
    fi
    for variant in "${variants[@]}"; do
      run_case_with_variant "target_single_1121__${variant}" "${variant}" \
        --case varlen --cu-seqlens 0,1121 --heads 8 --key-dim 128 --value-dim 128 --chunk-size 64 --tail-topk 16
    done
    ;;
  *)
    echo "unknown MODE=${MODE}; expected controls_and_target, controls, target, or target_isolated" >&2
    exit 2
    ;;
esac

SUMMARY_VARIANTS=${VARIANT_LIST:-${VARIANT}}
"${PYTHON}" - "${OUT_DIR}" "${SUMMARY_VARIANTS}" "${MODE}" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
variant_env = sys.argv[2]
mode = sys.argv[3]
all_case_files = {
    "fixed_1k_h8": out_dir / "fixed_1k_h8.json",
    "varlen_single_1024": out_dir / "varlen_single_1024.json",
    "varlen_aligned_1024": out_dir / "varlen_aligned_1024.json",
    "target_single_1121": out_dir / "target_single_1121.json",
}
if mode == "target":
    case_files = {"target_single_1121": all_case_files["target_single_1121"]}
elif mode == "target_isolated":
    case_files = {}
    isolated_variants = (
        variant_env.split()
        if variant_env != "all"
        else ["ascendc", "ascendc_saved_x", "manual_ascendc", "triton_full", "triton_dqkwg", "triton_wy", "triton_both", "triton_dvlocal"]
    )
    for variant in isolated_variants:
        case_files[f"target_single_1121__{variant}"] = out_dir / f"target_single_1121__{variant}.json"
elif mode == "controls":
    case_files = {name: all_case_files[name] for name in ("fixed_1k_h8", "varlen_single_1024", "varlen_aligned_1024")}
else:
    case_files = all_case_files
cases = {}
for name, path in case_files.items():
    if path.exists():
        cases[name] = json.loads(path.read_text())

if mode == "target_isolated":
    variants = ["__from_file__"]
elif variant_env == "all":
    variants = ["ascendc", "ascendc_saved_x", "manual_ascendc", "triton_full", "triton_dqkwg", "triton_wy", "triton_both", "triton_dvlocal"]
else:
    variants = [variant_env]
controls = ["fixed_1k_h8", "varlen_single_1024", "varlen_aligned_1024"]
target = "target_single_1121"

def variant_result(case_name, variant):
    if mode == "target_isolated" and case_name == target:
        case_name = f"{target}__{variant}"
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
    row_variants = list(cases.get(case_name, {}).get("variants", {})) if variants == ["__from_file__"] else variants
    display_case = case_name.split("__", 1)[0]
    for variant in row_variants:
        result = variant_result(case_name, variant)
        stats = grad_k_stats(result)
        tail = tail_stats(result)
        if result is None:
            print(f"{display_case:22s} {variant:14s} missing")
            continue
        if result.get("error"):
            first = result["error"].splitlines()[0] if result.get("error") else "error"
            print(f"{display_case:22s} {variant:14s} error    {'-':>10s} {'-':8s} {'-':>10s} {first}")
            continue
        print(
            f"{display_case:22s} {variant:14s} "
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
validity_variants = (
    [v for case in cases.values() for v in case.get("variants", {})]
    if variants == ["__from_file__"]
    else variants
)
for variant in validity_variants:
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
