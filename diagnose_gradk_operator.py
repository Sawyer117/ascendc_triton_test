#!/usr/bin/env python3
"""Localize varlen grad_k drift by replacing one backward branch with Triton.

The FLA-npu backward computes:

    grad_k = dk_from_dqkwg + dk_from_prepare_wy_repr_bwd

This diagnostic reuses the same inputs and PyTorch reference as
compare_gdn_precision.py, then runs four backward variants:

    ascendc:       original FLA-npu AscendC chain
    triton_dqkwg:  replace only npu_chunk_bwd_dqkwg
    triton_wy:     replace only npu_prepare_wy_repr_bwd_da/full
    triton_both:   replace both branches

The local Triton kernels live under ./local_triton and are used only by this
diagnostic script.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import compare_gdn_precision as cmp


VARIANTS: dict[str, tuple[bool, bool]] = {
    "ascendc": (False, False),
    "triton_dqkwg": (True, False),
    "triton_wy": (False, True),
    "triton_both": (True, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose which GDN backward branch causes grad_k precision drift."
    )
    parser.add_argument("--case", choices=cmp.CASES.keys(), default="varlen")
    parser.add_argument(
        "--fla-repo",
        default=os.environ.get("FLA_NPU_REPO"),
        help="Path to flash-linear-attention-npu. Defaults to ./flash-linear-attention-npu if present.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, help="Override batch size.")
    parser.add_argument("--seq-len", type=int, help="Override sequence length.")
    parser.add_argument("--heads", type=int, help="Override number of heads.")
    parser.add_argument("--key-dim", type=int, help="Override key/query head dim.")
    parser.add_argument("--value-dim", type=int, help="Override value head dim.")
    parser.add_argument("--chunk-size", type=int, help="Override chunk size.")
    parser.add_argument(
        "--cu-seqlens",
        type=str,
        help="Comma-separated cumulative sequence lengths. Batch must be 1.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--no-qk-l2norm", action="store_true")
    parser.add_argument("--output-json", type=str)
    parser.add_argument(
        "--variant",
        choices=[*VARIANTS.keys(), "all"],
        default="all",
        help="Run a single hybrid variant or all variants.",
    )
    return parser.parse_args()


def load_flash_module(fla_repo: Path):
    cmp.import_torch_runtime()
    cmp.check_npu_runtime()

    if str(fla_repo) not in sys.path:
        sys.path.insert(0, str(fla_repo))

    example_path = fla_repo / "examples" / "flash_gated_delta_rule.py"
    spec = importlib.util.spec_from_file_location("fla_npu_flash_gdn_diagnose", example_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {example_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_local_triton():
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    chunk_o = importlib.import_module("local_triton.chunk_o")
    wy_fast = importlib.import_module("local_triton.wy_fast")
    return chunk_o.chunk_bwd_dqkwg, wy_fast.prepare_wy_repr_bwd


def hf_to_tf(x):
    return x.transpose(1, 2).contiguous()


def tf_to_hf(x):
    return x.transpose(1, 2).contiguous()


def ensure_metadata(flash_module, g, cu, chunk_size: int):
    if cu is None:
        return None, None, None, None
    return flash_module._ensure_varlen_metadata(  # pylint: disable=protected-access
        g=g,
        cu_seqlens=cu,
        cu_seqlens_list=None,
        chunk_indices=None,
        chunk_indices_list=None,
        chunk_size=chunk_size,
    )


def chunk_list(flash_module, chunk_indices_list, chunk_size: int):
    return flash_module._chunk_list(chunk_indices_list, chunk_size)  # pylint: disable=protected-access


def run_reference(case: cmp.Case, args: argparse.Namespace, inputs: tuple[Any, ...], do):
    torch = cmp.torch
    q, k, v, beta, g, cu = inputs
    q_ref, k_ref, v_ref, beta_ref, g_ref = [cmp.clone_leaf(x) for x in (q, k, v, beta, g)]
    use_qk_l2norm = not args.no_qk_l2norm

    if cu is not None:
        q_ref_in, k_ref_in, v_ref_in, beta_ref_in, g_ref_in = cmp.varlen_to_nonvarlen(
            cu, q_ref, k_ref, v_ref, beta_ref, g_ref
        )
        for tensor in (q_ref_in, k_ref_in, v_ref_in, beta_ref_in, g_ref_in):
            tensor.retain_grad()
    else:
        q_ref_in, k_ref_in, v_ref_in, beta_ref_in, g_ref_in = q_ref, k_ref, v_ref, beta_ref, g_ref

    o_ref, _ = cmp.ref_torch_chunk_gated_delta_rule(
        q_ref_in,
        k_ref_in,
        v_ref_in,
        g_ref_in,
        beta_ref_in,
        chunk_size=case.chunk_size,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )
    do_ref = cmp.varlen_to_nonvarlen(cu, do) if cu is not None else do
    o_ref.backward(do_ref)
    torch.npu.synchronize()

    if cu is not None:
        grads = {
            "grad_q": q_ref_in.grad,
            "grad_k": k_ref_in.grad,
            "grad_v": v_ref_in.grad,
            "grad_beta": beta_ref_in.grad,
            "grad_g": g_ref_in.grad,
        }
    else:
        grads = {
            "grad_q": q_ref.grad,
            "grad_k": k_ref.grad,
            "grad_v": v_ref.grad,
            "grad_beta": beta_ref.grad,
            "grad_g": g_ref.grad,
        }
    return {"output": o_ref, **grads}


def run_backward_components(
    flash_module,
    triton_dqkwg,
    triton_wy,
    *,
    q,
    k,
    v,
    g,
    beta,
    A,
    scale: float,
    do,
    cu_seqlens,
    cu_seqlens_list,
    chunk_indices,
    chunk_indices_list,
    chunk_size: int,
    replace_dqkwg: bool,
    replace_wy: bool,
):
    torch = cmp.torch
    g_hf = g.transpose(1, 2).contiguous()
    beta_hf = beta.transpose(1, 2).contiguous().float()
    chunk_idx_list = chunk_list(flash_module, chunk_indices_list, chunk_size)

    w, u = torch.ops.npu.npu_recompute_w_u_fwd(
        k,
        v,
        beta_hf,
        A,
        chunk_size,
        g=g_hf,
        gk=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_idx_list,
    )

    do_hf = do.transpose(1, 2).contiguous()
    h, v_new, _ = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g_hf,
        gk=None,
        initial_state=None,
        output_final_state=False,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_idx_list,
        use_exp2=False,
        transpose_state_layout=False,
    )

    dv_local = torch.ops.npu.npu_chunk_bwd_dv_local(
        q,
        k,
        do_hf,
        g_hf,
        scale,
        chunk_size,
        g_gamma=None,
        A=A,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_idx_list,
    )

    dh, _, dv_mid = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
        q,
        k,
        w,
        do_hf,
        dv_local,
        scale,
        chunk_size,
        g=g_hf,
        gK=None,
        h0=None,
        dht=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=chunk_idx_list,
        use_exp2=False,
        transpose_state_layout=False,
    )

    if replace_dqkwg:
        dq_tf, dk_tf, dw_tf, dg_tf = triton_dqkwg(
            q=hf_to_tf(q),
            k=hf_to_tf(k),
            v=hf_to_tf(v_new),
            do=do,
            h=h.transpose(1, 2).contiguous(),
            dh=dh.transpose(1, 2).contiguous(),
            g=g,
            dv=hf_to_tf(dv_mid),
            w=hf_to_tf(w),
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            scale=scale,
        )
        dq = tf_to_hf(dq_tf)
        dk_from_dqkwg = tf_to_hf(dk_tf)
        dw = tf_to_hf(dw_tf)
        dg_from_dqkwg_tf = dg_tf
    else:
        dq, dk_from_dqkwg, dw, dg_hf = torch.ops.npu.npu_chunk_bwd_dqkwg(
            q,
            k,
            v_new,
            g_hf,
            h,
            do_hf,
            dh,
            dv_mid,
            chunk_size,
            cu_seqlens=cu_seqlens_list,
            chunk_indices=chunk_idx_list,
            w=None,
            g_gamma=None,
            scale=scale,
            use_exp2=False,
            transpose_state_layout=False,
        )
        dg_from_dqkwg_tf = dg_hf.transpose(1, 2).contiguous()

    if replace_wy:
        dk2_tf, dv_tf, db_tf, dg2_tf = triton_wy(
            k=hf_to_tf(k),
            v=hf_to_tf(v),
            beta=beta,
            g=g,
            A=A.transpose(1, 2).contiguous(),
            dw=hf_to_tf(dw),
            du=hf_to_tf(dv_mid),
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        dk_from_wy = tf_to_hf(dk2_tf)
        dv = tf_to_hf(dv_tf)
        db = db_tf
        dg2_tf = dg2_tf
    else:
        dA = torch.ops.npu.npu_prepare_wy_repr_bwd_da(
            k,
            v,
            beta_hf.float(),
            A,
            dw,
            dv_mid,
            g_hf.float(),
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens_list,
            chunk_indices=chunk_idx_list,
        )

        dk_from_wy, dv, db_hf, dg2_hf = torch.ops.npu.npu_prepare_wy_repr_bwd_full(
            k,
            v,
            beta_hf,
            A,
            dA,
            dw,
            dv_mid,
            g_hf,
            chunk_size,
            cu_seqlens=cu_seqlens_list,
            chunk_indices=chunk_idx_list,
        )
        db = db_hf.transpose(1, 2).contiguous()
        dg2_tf = dg2_hf.transpose(1, 2).contiguous()

    dk = dk_from_dqkwg + dk_from_wy
    dg = dg_from_dqkwg_tf + dg2_tf
    dg = flash_module.chunk_local_cumsum(
        dg,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=False,
    )
    return {
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "db": db,
        "dg": dg,
        "dk_from_dqkwg": dk_from_dqkwg,
        "dk_from_wy": dk_from_wy,
    }


def run_variant(
    name: str,
    case: cmp.Case,
    args: argparse.Namespace,
    flash_module,
    triton_dqkwg,
    triton_wy,
    inputs: tuple[Any, ...],
    do,
):
    torch = cmp.torch
    q, k, v, beta, g, cu = inputs
    replace_dqkwg, replace_wy = VARIANTS[name]
    use_qk_l2norm = not args.no_qk_l2norm

    q_hf = q.transpose(1, 2).contiguous()
    k_hf = k.transpose(1, 2).contiguous()
    v_hf = v.transpose(1, 2).contiguous()

    if use_qk_l2norm:
        q_work, q_rstd = flash_module.l2norm_fwd(q_hf)
        k_work, k_rstd = flash_module.l2norm_fwd(k_hf)
    else:
        q_work, k_work = q_hf, k_hf
        q_rstd, k_rstd = None, None

    cu, cu_list, chunk_indices, chunk_indices_list = ensure_metadata(
        flash_module, g, cu, case.chunk_size
    )
    scale = case.key_dim ** -0.5
    g_cum, output, A, _ = flash_module.flash_chunk_gated_delta_rule_fwd(
        q=q_work,
        k=k_work,
        v=v_hf,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu,
        cu_seqlens_list=cu_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        chunk_size=case.chunk_size,
    )

    parts = run_backward_components(
        flash_module,
        triton_dqkwg,
        triton_wy,
        q=q_work,
        k=k_work,
        v=v_hf,
        g=g_cum,
        beta=beta,
        A=A,
        scale=scale,
        do=do,
        cu_seqlens=cu,
        cu_seqlens_list=cu_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        chunk_size=case.chunk_size,
        replace_dqkwg=replace_dqkwg,
        replace_wy=replace_wy,
    )

    dq = parts["dq"]
    dk = parts["dk"]
    if use_qk_l2norm:
        dq = flash_module.l2norm_bwd(q_work, q_rstd, dq)
        dk = flash_module.l2norm_bwd(k_work, k_rstd, dk)

    torch.npu.synchronize()
    return {
        "output": output,
        "grad_q": hf_to_tf(dq),
        "grad_k": hf_to_tf(dk),
        "grad_v": hf_to_tf(parts["dv"]),
        "grad_beta": parts["db"],
        "grad_g": parts["dg"],
        "components": {
            "dk_from_dqkwg": hf_to_tf(parts["dk_from_dqkwg"]),
            "dk_from_wy": hf_to_tf(parts["dk_from_wy"]),
        },
    }


def maybe_unflatten(cu, *tensors):
    if cu is None:
        return tensors[0] if len(tensors) == 1 else tensors
    return cmp.varlen_to_nonvarlen(cu, *tensors)


def compare_outputs(case_result: dict[str, Any], reference: dict[str, Any], cu, args: argparse.Namespace):
    device = case_result["grad_k"].device
    valid_mask = cmp.varlen_valid_mask(cu, device) if cu is not None else None
    comparisons = {}
    for name in ("output", "grad_q", "grad_k", "grad_v", "grad_beta", "grad_g"):
        actual = maybe_unflatten(cu, case_result[name])
        expected = reference[name]
        comparisons[name] = cmp.tensor_stats(actual, expected, args.atol, args.rtol, valid_mask)
    return comparisons


def component_delta(left: dict[str, Any], right: dict[str, Any], cu, args: argparse.Namespace):
    device = left["components"]["dk_from_dqkwg"].device
    valid_mask = cmp.varlen_valid_mask(cu, device) if cu is not None else None
    out = {}
    for name in ("dk_from_dqkwg", "dk_from_wy"):
        lhs = maybe_unflatten(cu, left["components"][name])
        rhs = maybe_unflatten(cu, right["components"][name])
        out[name] = cmp.tensor_stats(lhs, rhs, args.atol, args.rtol, valid_mask)
    return out


def failed_tensors(comparisons: dict[str, Any]) -> list[str]:
    return [name for name, stats in comparisons.items() if not stats["allclose"]]


def infer_culprit(variant_results: dict[str, Any]) -> str:
    present = set(variant_results)
    if "ascendc" not in present:
        return "Run the ascendc variant to infer a culprit."
    if variant_results["ascendc"]["comparisons"]["grad_k"]["allclose"]:
        return "ascendc grad_k already passes; no failing branch in this case."

    dqkwg_ok = (
        "triton_dqkwg" in present
        and variant_results["triton_dqkwg"]["comparisons"]["grad_k"]["allclose"]
    )
    wy_ok = (
        "triton_wy" in present
        and variant_results["triton_wy"]["comparisons"]["grad_k"]["allclose"]
    )
    both_ok = (
        "triton_both" in present
        and variant_results["triton_both"]["comparisons"]["grad_k"]["allclose"]
    )
    if dqkwg_ok and not wy_ok:
        return "Most likely culprit: npu_chunk_bwd_dqkwg."
    if wy_ok and not dqkwg_ok:
        return "Most likely culprit: npu_prepare_wy_repr_bwd_da/full."
    if dqkwg_ok and wy_ok:
        return "Both single replacements make grad_k pass; inspect component deltas for the primary drift."
    if both_ok:
        return "Only replacing both branches makes grad_k pass; the error is split across dqkwg and WY backward."
    return "No Triton replacement fixed grad_k; check earlier backward inputs or Triton layout compatibility."


def segment_summary(cu_seqlens: tuple[int, ...] | None, total_tokens: int, chunk_size: int):
    if cu_seqlens is None:
        lengths = [total_tokens]
    else:
        lengths = [b - a for a, b in zip(cu_seqlens, cu_seqlens[1:])]
    return {
        "total_tokens": total_tokens,
        "chunk_size": chunk_size,
        "total_tail": total_tokens % chunk_size,
        "segment_lengths": lengths,
        "segment_tails": [length % chunk_size for length in lengths],
    }


def main() -> int:
    args = parse_args()
    try:
        torch, _ = cmp.import_torch_runtime()
        fla_repo = cmp.find_fla_repo(args.fla_repo)
        flash_module = load_flash_module(fla_repo)

        device = torch.device(f"npu:{args.device}")
        torch.npu.set_device(args.device)
        if hasattr(torch.npu, "set_compile_mode"):
            torch.npu.set_compile_mode(jit_compile=False)

        triton_dqkwg, triton_wy = load_local_triton()
        case = cmp.override_case(cmp.CASES[args.case], args)
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        inputs = cmp.make_inputs(case, device, dtype, args.seed)

        torch.manual_seed(args.seed + 1)
        do = torch.randn(
            case.batch,
            case.seq_len,
            case.heads,
            case.value_dim,
            dtype=dtype,
            device=device,
        )
        reference = run_reference(case, args, inputs, do)

        names = list(VARIANTS) if args.variant == "all" else [args.variant]
        raw_results = {}
        variant_results = {}
        for name in names:
            print(f"==> {name}", flush=True)
            raw = run_variant(
                name,
                case,
                args,
                flash_module,
                triton_dqkwg,
                triton_wy,
                inputs,
                do,
            )
            comparisons = compare_outputs(raw, reference, inputs[-1], args)
            raw_results[name] = raw
            variant_results[name] = {
                "replace_dqkwg": VARIANTS[name][0],
                "replace_wy": VARIANTS[name][1],
                "passed": all(stats["allclose"] for stats in comparisons.values()),
                "failed_tensors": failed_tensors(comparisons),
                "comparisons": comparisons,
            }
            grad_k = comparisons["grad_k"]
            print(
                "    grad_k",
                f"allclose={grad_k['allclose']}",
                f"max_abs={grad_k['max_abs']:.6g}",
                f"rms={grad_k['rms']:.6g}",
                f"mismatch={grad_k['mismatch_ratio']:.6g}",
                flush=True,
            )

        component_deltas = {}
        if "ascendc" in raw_results and "triton_both" in raw_results:
            component_deltas["ascendc_vs_triton_both"] = component_delta(
                raw_results["ascendc"], raw_results["triton_both"], inputs[-1], args
            )

        conclusion = infer_culprit(variant_results)
        payload = {
            "case": case.__dict__,
            "dtype": args.dtype,
            "device": str(device),
            "seed": args.seed,
            "atol": args.atol,
            "rtol": args.rtol,
            "fla_repo": str(fla_repo),
            "torch": torch.__version__,
            "shape_summary": segment_summary(case.cu_seqlens, case.seq_len, case.chunk_size),
            "variants": variant_results,
            "component_deltas": component_deltas,
            "conclusion": conclusion,
        }

        print("\nsummary")
        for name, result in variant_results.items():
            gk = result["comparisons"]["grad_k"]
            print(
                f"{name:14s} passed={result['passed']} "
                f"grad_k_allclose={gk['allclose']} "
                f"max_abs={gk['max_abs']:.6g} "
                f"rms={gk['rms']:.6g} "
                f"mismatch={gk['mismatch_ratio']:.6g} "
                f"failed={','.join(result['failed_tensors']) or '-'}"
            )
        if component_deltas:
            print("\ncomponent deltas: ascendc vs triton_both")
            for name, stats in component_deltas["ascendc_vs_triton_both"].items():
                print(
                    f"{name:14s} max_abs={stats['max_abs']:.6g} "
                    f"rms={stats['rms']:.6g} mismatch={stats['mismatch_ratio']:.6g}"
                )
        print(f"\nconclusion: {conclusion}")

        text = json.dumps(payload, indent=2, sort_keys=True)
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                f.write(text)
                f.write("\n")
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
