#!/usr/bin/env python3
"""Trace GDN varlen grad_k drift by comparing intermediate ops.

This script avoids the fragile hybrid-replacement conclusion path.  It runs
AscendC and local Triton implementations on the same intermediate inputs, then
reports the first component whose output diverges, with tail stats for
time-major tensors.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import compare_gdn_precision as cmp
import diagnose_gradk_operator as diag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace GDN intermediate op precision.")
    parser.add_argument("--case", choices=cmp.CASES.keys(), default="varlen")
    parser.add_argument("--fla-repo", default=os.environ.get("FLA_NPU_REPO"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--key-dim", type=int)
    parser.add_argument("--value-dim", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--cu-seqlens", type=str)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--no-qk-l2norm", action="store_true")
    parser.add_argument("--tail-topk", type=int, default=8)
    parser.add_argument("--output-json", type=str)
    return parser.parse_args()


def load_local_modules():
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return {
        "cumsum": importlib.import_module("local_triton.cumsum"),
        "kkt": importlib.import_module("local_triton.chunk_scaled_dot_kkt"),
        "solve": importlib.import_module("local_triton.solve_tril"),
        "wy": importlib.import_module("local_triton.wy_fast"),
        "delta_h": importlib.import_module("local_triton.chunk_delta_h"),
        "chunk_o": importlib.import_module("local_triton.chunk_o"),
    }


def add_nan_counts(stats: dict[str, Any], actual, expected) -> dict[str, Any]:
    torch = cmp.torch
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    stats = dict(stats)
    stats["actual_nan"] = int(torch.isnan(actual_f).sum().item())
    stats["expected_nan"] = int(torch.isnan(expected_f).sum().item())
    stats["actual_inf"] = int(torch.isinf(actual_f).sum().item())
    stats["expected_inf"] = int(torch.isinf(expected_f).sum().item())
    return stats


def compare_time_tensor(
    name: str,
    actual,
    expected,
    *,
    case: cmp.Case,
    cu,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = actual.device
    valid_mask = cmp.varlen_valid_mask(cu, device) if cu is not None else None
    actual_cmp = diag.maybe_unflatten(cu, actual)
    expected_cmp = diag.maybe_unflatten(cu, expected)
    stats = cmp.tensor_stats(actual_cmp, expected_cmp, args.atol, args.rtol, valid_mask)
    stats = add_nan_counts(stats, actual_cmp, expected_cmp)

    result: dict[str, Any] = {
        "kind": "time",
        "shape": list(actual.shape),
        "all": stats,
    }

    tail_mask = diag.sequence_partial_tail_mask(cu, device, case.chunk_size)
    if tail_mask is None and case.seq_len % case.chunk_size:
        tail_mask = diag.mask_packed_range(
            case,
            cu,
            device,
            case.seq_len - (case.seq_len % case.chunk_size),
            case.seq_len,
        )
    if tail_mask is not None and bool(tail_mask.any().item()):
        tail_stats = cmp.tensor_stats(actual_cmp, expected_cmp, args.atol, args.rtol, tail_mask)
        tail_stats = add_nan_counts(tail_stats, actual_cmp, expected_cmp)
        result["tail"] = tail_stats
        result["tail_top_errors"] = diag.topk_errors(
            actual_cmp,
            expected_cmp,
            tail_mask,
            cu,
            case,
            args.tail_topk,
        )
    return result


def compare_global_tensor(name: str, actual, expected, *, args: argparse.Namespace) -> dict[str, Any]:
    stats = cmp.tensor_stats(actual, expected, args.atol, args.rtol)
    stats = add_nan_counts(stats, actual, expected)
    return {
        "kind": "global",
        "shape": list(actual.shape),
        "all": stats,
    }


def compare_component(
    results: dict[str, Any],
    name: str,
    actual,
    expected,
    *,
    case: cmp.Case,
    cu,
    args: argparse.Namespace,
    time_major: bool = True,
) -> None:
    if list(actual.shape) != list(expected.shape):
        results[name] = {
            "kind": "shape_mismatch",
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "all": {"allclose": False},
        }
        return
    if time_major:
        results[name] = compare_time_tensor(name, actual, expected, case=case, cu=cu, args=args)
    else:
        results[name] = compare_global_tensor(name, actual, expected, args=args)


def print_component_summary(results: dict[str, Any]) -> None:
    print("\ncomponent summary")
    for name, item in results.items():
        stats = item["all"]
        tail = item.get("tail")
        tail_text = "-"
        if tail is not None:
            tail_text = (
                f"{tail['allclose']} max_abs={tail['max_abs']:.6g} "
                f"rms={tail['rms']:.6g}"
            )
        if "max_abs" in stats:
            print(
                f"{name:24s} allclose={stats['allclose']} "
                f"max_abs={stats['max_abs']:.6g} rms={stats['rms']:.6g} "
                f"nan={stats['actual_nan']}/{stats['expected_nan']} "
                f"tail={tail_text}"
            )
        else:
            print(f"{name:24s} allclose=False shape_mismatch")


def first_failed(results: dict[str, Any]) -> str | None:
    for name, item in results.items():
        if not item.get("all", {}).get("allclose", False):
            return name
    return None


def main() -> int:
    args = parse_args()
    try:
        torch, _ = cmp.import_torch_runtime()
        fla_repo = cmp.find_fla_repo(args.fla_repo)
        flash_module = diag.load_flash_module(fla_repo)
        local = load_local_modules()

        device = torch.device(f"npu:{args.device}")
        torch.npu.set_device(args.device)
        if hasattr(torch.npu, "set_compile_mode"):
            torch.npu.set_compile_mode(jit_compile=False)

        case = cmp.override_case(cmp.CASES[args.case], args)
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        q, k, v, beta, g, cu = cmp.make_inputs(case, device, dtype, args.seed)
        torch.manual_seed(args.seed + 1)
        do = torch.randn(
            case.batch,
            case.seq_len,
            case.heads,
            case.value_dim,
            dtype=dtype,
            device=device,
        )

        q_hf = q.transpose(1, 2).contiguous()
        k_hf = k.transpose(1, 2).contiguous()
        v_hf = v.transpose(1, 2).contiguous()
        use_l2norm = not args.no_qk_l2norm
        if use_l2norm:
            q_work_hf = cmp.l2norm(q_hf, dim=-1, eps=1e-6).to(dtype)
            k_work_hf = cmp.l2norm(k_hf, dim=-1, eps=1e-6).to(dtype)
        else:
            q_work_hf, k_work_hf = q_hf, k_hf

        cu_meta, cu_list, chunk_indices, chunk_indices_list = diag.ensure_metadata(
            flash_module, g, cu, case.chunk_size
        )
        chunk_idx_list = diag.chunk_list(flash_module, chunk_indices_list, case.chunk_size)
        scale = case.key_dim ** -0.5

        print(
            "running trace",
            f"case={case.name}",
            f"T={case.seq_len}",
            f"H={case.heads}",
            f"K={case.key_dim}",
            f"V={case.value_dim}",
            f"chunk={case.chunk_size}",
            f"varlen={cu is not None}",
            f"l2norm={use_l2norm}",
            flush=True,
        )

        # AscendC wrapper-side forward intermediates.
        g_ac, o_ac, A_ac, _ = flash_module.flash_chunk_gated_delta_rule_fwd(
            q=q_work_hf,
            k=k_work_hf,
            v=v_hf,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_meta,
            cu_seqlens_list=cu_list,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_indices_list,
            chunk_size=case.chunk_size,
        )

        g_ac_hf = g_ac.transpose(1, 2).contiguous()
        beta_hf = beta.transpose(1, 2).contiguous().float()
        w_ac, u_ac = torch.ops.npu.npu_recompute_w_u_fwd(
            k_work_hf,
            v_hf,
            beta_hf,
            A_ac,
            case.chunk_size,
            g=g_ac_hf,
            gk=None,
            cu_seqlens=cu_list,
            chunk_indices=chunk_idx_list,
        )
        h_ac, v_new_ac, _ = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
            k_work_hf,
            w_ac,
            u_ac,
            g=g_ac_hf,
            gk=None,
            initial_state=None,
            output_final_state=False,
            chunk_size=case.chunk_size,
            save_new_value=True,
            cu_seqlens=cu_list,
            chunk_indices=chunk_idx_list,
            use_exp2=False,
            transpose_state_layout=False,
        )

        # Local Triton forward intermediates, same normalized q/k contract.
        q_work_tf = diag.hf_to_tf(q_work_hf)
        k_work_tf = diag.hf_to_tf(k_work_hf)
        g_tri = local["cumsum"].chunk_local_cumsum(
            g,
            chunk_size=case.chunk_size,
            cu_seqlens=cu_meta,
            head_first=False,
        )
        A_tri = local["kkt"].chunk_scaled_dot_kkt_fwd(
            k=k_work_tf,
            g=g_tri,
            beta=beta,
            cu_seqlens=cu_meta,
            chunk_size=case.chunk_size,
            output_dtype=torch.float32,
        )
        A_tri = local["solve"].solve_tril(A=A_tri, cu_seqlens=cu_meta, output_dtype=k_work_tf.dtype)
        w_tri_tf, u_tri_tf = local["wy"].recompute_w_u_fwd(
            k=k_work_tf,
            v=v,
            beta=beta,
            A=A_tri,
            g=g_tri,
            cu_seqlens=cu_meta,
        )
        h_tri, v_new_tri_tf, _ = local["delta_h"].chunk_gated_delta_rule_fwd_h(
            k=k_work_tf,
            w=w_tri_tf,
            u=u_tri_tf,
            g=g_tri,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_meta,
            chunk_size=case.chunk_size,
        )
        o_tri = local["chunk_o"].chunk_fwd_o(
            q=q_work_tf,
            k=k_work_tf,
            v=v_new_tri_tf,
            h=h_tri,
            g=g_tri,
            scale=scale,
            cu_seqlens=cu_meta,
            chunk_size=case.chunk_size,
        )

        results: dict[str, Any] = {}
        compare_component(results, "fwd.g_cumsum", g_ac, g_tri, case=case, cu=cu, args=args)
        A_ac_tf = A_ac.transpose(1, 2).contiguous()
        compare_component(results, "fwd.A_solved", A_ac_tf, A_tri, case=case, cu=cu, args=args)
        compare_component(results, "fwd.w", diag.hf_to_tf(w_ac), w_tri_tf, case=case, cu=cu, args=args)
        compare_component(results, "fwd.u", diag.hf_to_tf(u_ac), u_tri_tf, case=case, cu=cu, args=args)
        compare_component(results, "fwd.h", h_ac, h_tri, case=case, cu=cu, args=args, time_major=False)
        compare_component(results, "fwd.v_new", diag.hf_to_tf(v_new_ac), v_new_tri_tf, case=case, cu=cu, args=args)
        compare_component(results, "fwd.output", o_ac, o_tri, case=case, cu=cu, args=args)

        do_hf = do.transpose(1, 2).contiguous()
        dv_local_ac = torch.ops.npu.npu_chunk_bwd_dv_local(
            q_work_hf,
            k_work_hf,
            do_hf,
            g_ac_hf,
            scale,
            case.chunk_size,
            g_gamma=None,
            A=A_ac,
            cu_seqlens=cu_list,
            chunk_indices=chunk_idx_list,
        )
        dv_local_tri_tf = local["chunk_o"].chunk_bwd_dv_local(
            q=q_work_tf,
            k=k_work_tf,
            do=do,
            g=g_ac,
            scale=scale,
            cu_seqlens=cu_meta,
            chunk_size=case.chunk_size,
        )
        compare_component(
            results,
            "bwd.dv_local",
            diag.hf_to_tf(dv_local_ac),
            dv_local_tri_tf,
            case=case,
            cu=cu,
            args=args,
        )

        dh_ac, _, dv_mid_ac = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
            q_work_hf,
            k_work_hf,
            w_ac,
            do_hf,
            dv_local_ac,
            scale,
            case.chunk_size,
            g=g_ac_hf,
            gK=None,
            h0=None,
            dht=None,
            cu_seqlens=cu_list,
            chunk_indices=chunk_idx_list,
            use_exp2=False,
            transpose_state_layout=False,
        )
        dh_tri, _, dv_mid_tri_tf = local["delta_h"].chunk_gated_delta_rule_bwd_dhu(
            q=q_work_tf,
            k=k_work_tf,
            w=diag.hf_to_tf(w_ac),
            do=do,
            dv=diag.hf_to_tf(dv_local_ac),
            g=g_ac,
            h0=None,
            dht=None,
            scale=scale,
            cu_seqlens=cu_meta,
            chunk_size=case.chunk_size,
        )
        compare_component(results, "bwd.dhu.dh", dh_ac, dh_tri, case=case, cu=cu, args=args, time_major=False)
        compare_component(
            results,
            "bwd.dhu.dv_mid",
            diag.hf_to_tf(dv_mid_ac),
            dv_mid_tri_tf,
            case=case,
            cu=cu,
            args=args,
        )

        dq_ac, dk_dqkwg_ac, dw_ac, dg_dqkwg_hf = torch.ops.npu.npu_chunk_bwd_dqkwg(
            q_work_hf,
            k_work_hf,
            v_new_ac,
            g_ac_hf,
            h_ac,
            do_hf,
            dh_ac,
            dv_mid_ac,
            case.chunk_size,
            cu_seqlens=cu_list,
            chunk_indices=chunk_idx_list,
            w=None,
            g_gamma=None,
            scale=scale,
            use_exp2=False,
            transpose_state_layout=False,
        )
        dq_tri_tf, dk_dqkwg_tri_tf, dw_tri_tf, dg_dqkwg_tri = local["chunk_o"].chunk_bwd_dqkwg(
            q=q_work_tf,
            k=k_work_tf,
            v=diag.hf_to_tf(v_new_ac),
            do=do,
            h=h_ac.transpose(1, 2).contiguous(),
            dh=dh_ac.transpose(1, 2).contiguous(),
            g=g_ac,
            dv=diag.hf_to_tf(dv_mid_ac),
            w=diag.hf_to_tf(w_ac),
            cu_seqlens=cu_meta,
            chunk_size=case.chunk_size,
            scale=scale,
        )
        dg_dqkwg_ac = dg_dqkwg_hf.transpose(1, 2).contiguous()
        compare_component(results, "bwd.dqkwg.dq", diag.hf_to_tf(dq_ac), dq_tri_tf, case=case, cu=cu, args=args)
        compare_component(results, "bwd.dqkwg.dk", diag.hf_to_tf(dk_dqkwg_ac), dk_dqkwg_tri_tf, case=case, cu=cu, args=args)
        compare_component(results, "bwd.dqkwg.dw", diag.hf_to_tf(dw_ac), dw_tri_tf, case=case, cu=cu, args=args)
        compare_component(results, "bwd.dqkwg.dg", dg_dqkwg_ac, dg_dqkwg_tri, case=case, cu=cu, args=args)

        dA_ac = torch.ops.npu.npu_prepare_wy_repr_bwd_da(
            k_work_hf,
            v_hf,
            beta_hf.float(),
            A_ac,
            dw_ac,
            dv_mid_ac,
            g_ac_hf.float(),
            chunk_size=case.chunk_size,
            cu_seqlens=cu_list,
            chunk_indices=chunk_idx_list,
        )
        dk_wy_ac, dv_wy_ac, db_hf_ac, dg_wy_hf_ac = torch.ops.npu.npu_prepare_wy_repr_bwd_full(
            k_work_hf,
            v_hf,
            beta_hf,
            A_ac,
            dA_ac,
            dw_ac,
            dv_mid_ac,
            g_ac_hf,
            case.chunk_size,
            cu_seqlens=cu_list,
            chunk_indices=chunk_idx_list,
        )
        dk_wy_tri_tf, dv_wy_tri_tf, db_tri, dg_wy_tri = local["wy"].prepare_wy_repr_bwd(
            k=k_work_tf,
            v=v,
            beta=beta,
            g=g_ac,
            A=A_ac_tf,
            dw=diag.hf_to_tf(dw_ac),
            du=diag.hf_to_tf(dv_mid_ac),
            cu_seqlens=cu_meta,
            chunk_size=case.chunk_size,
        )
        db_ac = db_hf_ac.transpose(1, 2).contiguous()
        dg_wy_ac = dg_wy_hf_ac.transpose(1, 2).contiguous()
        compare_component(results, "bwd.wy.dk", diag.hf_to_tf(dk_wy_ac), dk_wy_tri_tf, case=case, cu=cu, args=args)
        compare_component(results, "bwd.wy.dv", diag.hf_to_tf(dv_wy_ac), dv_wy_tri_tf, case=case, cu=cu, args=args)
        compare_component(results, "bwd.wy.dbeta", db_ac, db_tri, case=case, cu=cu, args=args)
        compare_component(results, "bwd.wy.dg", dg_wy_ac, dg_wy_tri, case=case, cu=cu, args=args)
        compare_component(
            results,
            "bwd.dk_core_sum",
            diag.hf_to_tf(dk_dqkwg_ac + dk_wy_ac),
            dk_dqkwg_tri_tf + dk_wy_tri_tf,
            case=case,
            cu=cu,
            args=args,
        )

        print_component_summary(results)
        first = first_failed(results)
        if first:
            print(f"\nfirst_failed_component: {first}")
        else:
            print("\nfirst_failed_component: none")

        payload = {
            "case": case.__dict__,
            "dtype": args.dtype,
            "device": str(device),
            "seed": args.seed,
            "atol": args.atol,
            "rtol": args.rtol,
            "fla_repo": str(fla_repo),
            "torch": torch.__version__,
            "shape_summary": diag.segment_summary(case.cu_seqlens, case.seq_len, case.chunk_size),
            "use_qk_l2norm_in_kernel": use_l2norm,
            "first_failed_component": first,
            "components": results,
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                f.write(text)
                f.write("\n")
        return 1 if first else 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
