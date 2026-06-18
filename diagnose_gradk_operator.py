#!/usr/bin/env python3
"""Localize varlen grad_k drift by replacing one backward branch with Triton.

The FLA-npu backward computes:

    grad_k = dk_from_dqkwg + dk_from_prepare_wy_repr_bwd

This diagnostic reuses the same inputs and PyTorch reference as
compare_gdn_precision.py, then runs targeted backward variants:

    ascendc:       original FLA-npu AscendC chain
    ascendc_saved_x: same wrapper, but l2norm_bwd receives original q/k
    triton_dhu:    replace only npu_chunk_gated_delta_rule_bwd_dhu
    triton_dqkwg:  replace only npu_chunk_bwd_dqkwg
    triton_wy:     replace only npu_prepare_wy_repr_bwd_da/full
    triton_both:   replace dqkwg and WY branches
    triton_all:    replace dhu, dqkwg, and WY branches

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


# Tuple layout: replace_dv_local, replace_dhu, replace_dqkwg, replace_wy.
VARIANTS: dict[str, tuple[bool, bool, bool, bool]] = {
    "ascendc": (False, False, False, False),
    "ascendc_saved_x": (False, False, False, False),
    "ascendc_py_l2norm_norm": (False, False, False, False),
    "ascendc_py_l2norm_orig": (False, False, False, False),
    "manual_ascendc": (False, False, False, False),
    "triton_full": (False, False, False, False),
    "triton_dqkwg": (False, False, True, False),
    "triton_wy": (False, False, False, True),
    "triton_both": (False, False, True, True),
    # Keep dv_local late in --variant all because this experimental hybrid can perturb later NPU work.
    "triton_dvlocal": (True, False, False, False),
    "triton_dhu": (False, True, False, False),
    "triton_dvlocal_dhu": (True, True, False, False),
    "triton_dhu_dqkwg": (False, True, True, False),
    "triton_all": (True, True, True, True),
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
    parser.add_argument(
        "--manual-l2norm-bwd-input",
        choices=["normalized", "original"],
        default=os.environ.get("MANUAL_L2NORM_BWD_INPUT", "normalized"),
        help=(
            "Input passed to l2norm_bwd in manual/hybrid chains. "
            "normalized preserves old wrapper parity; original tests the saved-input contract."
        ),
    )
    parser.add_argument("--output-json", type=str)
    parser.add_argument(
        "--tail-topk",
        type=int,
        default=8,
        help="Number of largest grad_k tail errors to print and write to JSON.",
    )
    parser.add_argument(
        "--include-dhu-hybrid",
        action="store_true",
        help="Include experimental bwd_dhu Triton hybrid variants when --variant all is used.",
    )
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


def load_full_triton_gdn():
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    module = importlib.import_module("local_triton_gdn")
    return module.chunk_gated_delta_rule


def load_local_triton():
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    chunk_delta_h = importlib.import_module("local_triton.chunk_delta_h")
    chunk_o = importlib.import_module("local_triton.chunk_o")
    wy_fast = importlib.import_module("local_triton.wy_fast")
    return (
        chunk_o.chunk_bwd_dv_local,
        chunk_delta_h.chunk_gated_delta_rule_bwd_dhu,
        chunk_o.chunk_bwd_dqkwg,
        wy_fast.prepare_wy_repr_bwd,
    )


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
    triton_dvlocal,
    triton_dhu,
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
    replace_dv_local: bool,
    replace_dhu: bool,
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

    if replace_dv_local:
        dv_local_tf = triton_dvlocal(
            q=hf_to_tf(q),
            k=hf_to_tf(k),
            do=do,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        dv_local = tf_to_hf(dv_local_tf)
    else:
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

    if replace_dhu:
        dh_tf, _, dv_mid_tf = triton_dhu(
            q=hf_to_tf(q),
            k=hf_to_tf(k),
            w=hf_to_tf(w),
            do=do,
            dv=hf_to_tf(dv_local),
            g=g,
            h0=None,
            dht=None,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        dh = dh_tf.transpose(1, 2).contiguous()
        dv_mid = tf_to_hf(dv_mid_tf)
    else:
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
        "dv_local": dv_local,
        "dv_mid": dv_mid,
        "dh": dh,
    }


def run_full_triton_variant(case: cmp.Case, args: argparse.Namespace, full_triton_gdn, inputs: tuple[Any, ...], do):
    torch = cmp.torch
    q, k, v, beta, g, cu = inputs
    q_tri, k_tri, v_tri, beta_tri, g_tri = [cmp.clone_leaf(x) for x in (q, k, v, beta, g)]
    use_qk_l2norm = not args.no_qk_l2norm

    o_tri, _ = full_triton_gdn(
        q=q_tri,
        k=k_tri,
        v=v_tri,
        g=g_tri,
        beta=beta_tri,
        cu_seqlens=cu,
        chunk_size=case.chunk_size,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )
    o_tri.backward(do)
    torch.npu.synchronize()
    return {
        "output": o_tri,
        "grad_q": q_tri.grad,
        "grad_k": k_tri.grad,
        "grad_v": v_tri.grad,
        "grad_beta": beta_tri.grad,
        "grad_g": g_tri.grad,
        "components": {},
    }


def run_wrapper_ascendc_variant(case: cmp.Case, args: argparse.Namespace, flash_module, inputs: tuple[Any, ...], do):
    torch = cmp.torch
    q, k, v, beta, g, cu = inputs
    q_ac, k_ac, v_ac = [cmp.clone_leaf(x.transpose(1, 2).contiguous()) for x in (q, k, v)]
    beta_ac, g_ac = [cmp.clone_leaf(x) for x in (beta, g)]
    use_qk_l2norm = not args.no_qk_l2norm

    o_ac, _ = flash_module.flash_gated_delta_rule(
        q=q_ac,
        k=k_ac,
        v=v_ac,
        g=g_ac,
        beta=beta_ac,
        cu_seqlens=cu,
        chunk_size=case.chunk_size,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )
    o_ac.backward(do)
    torch.npu.synchronize()
    return {
        "output": o_ac,
        "grad_q": q_ac.grad.transpose(1, 2).contiguous(),
        "grad_k": k_ac.grad.transpose(1, 2).contiguous(),
        "grad_v": v_ac.grad.transpose(1, 2).contiguous(),
        "grad_beta": beta_ac.grad,
        "grad_g": g_ac.grad,
        "components": {},
    }


def _broadcast_rstd(rstd, target):
    if rstd.ndim == target.ndim - 1:
        return rstd.unsqueeze(-1)
    return rstd


def py_l2norm_bwd_normalized(y, rstd, dy):
    rstd = _broadcast_rstd(rstd, dy)
    dot = (dy * y).sum(dim=-1, keepdim=True)
    return ((dy - y * dot) * rstd).to(y.dtype)


def py_l2norm_bwd_original(x, rstd, dy):
    rstd = _broadcast_rstd(rstd, dy)
    y = (x * rstd).to(x.dtype)
    dot = (dy * y).sum(dim=-1, keepdim=True)
    return ((dy - y * dot) * rstd).to(x.dtype)


def run_wrapper_l2norm_bwd_variant(
    case: cmp.Case,
    args: argparse.Namespace,
    flash_module,
    inputs: tuple[Any, ...],
    do,
    l2norm_bwd_mode: str,
):
    """Run the real AscendC wrapper path with a selected final l2norm backward."""
    torch = cmp.torch
    q, k, v, beta, g, cu = inputs
    q_ac, k_ac, v_ac = [cmp.clone_leaf(x.transpose(1, 2).contiguous()) for x in (q, k, v)]
    beta_ac, g_ac = [cmp.clone_leaf(x) for x in (beta, g)]
    use_qk_l2norm = not args.no_qk_l2norm

    class SavedXL2NormFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            q_in,
            k_in,
            v_in,
            g_in,
            beta_in,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            cu_seqlens_list,
            chunk_indices,
            chunk_indices_list,
            use_l2norm,
            chunk_size,
        ):
            q_orig = q_in
            k_orig = k_in
            if use_l2norm:
                q_work, q_rstd = flash_module.l2norm_fwd(q_in)
                k_work, k_rstd = flash_module.l2norm_fwd(k_in)
            else:
                q_work, k_work = q_in, k_in
                q_rstd = q_in.new_empty(0)
                k_rstd = k_in.new_empty(0)

            g_out, o, A, final_state = flash_module.flash_chunk_gated_delta_rule_fwd(
                q=q_work,
                k=k_work,
                v=v_in,
                g=g_in,
                beta=beta_in,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                cu_seqlens_list=cu_seqlens_list,
                chunk_indices=chunk_indices,
                chunk_indices_list=chunk_indices_list,
                chunk_size=chunk_size,
            )
            ctx.save_for_backward(q_orig, k_orig, q_work, k_work, q_rstd, k_rstd, v_in, g_out, beta_in, A)
            ctx.scale = scale
            ctx.initial_state = initial_state
            ctx.cu_seqlens = cu_seqlens
            ctx.cu_seqlens_list = cu_seqlens_list
            ctx.chunk_indices = chunk_indices
            ctx.chunk_indices_list = chunk_indices_list
            ctx.use_l2norm = use_l2norm
            ctx.chunk_size = chunk_size
            return o.to(q_in.dtype), final_state

        @staticmethod
        def backward(ctx, do_grad, dht):
            q_orig, k_orig, q_work, k_work, q_rstd, k_rstd, v_in, g_out, beta_in, A = ctx.saved_tensors
            dq, dk, dv, db, dg, dh0 = flash_module.flash_chunk_gated_delta_rule_bwd(
                q=q_work,
                k=k_work,
                v=v_in,
                g=g_out,
                beta=beta_in,
                A=A,
                scale=ctx.scale,
                initial_state=ctx.initial_state,
                do=do_grad,
                dht=dht,
                cu_seqlens=ctx.cu_seqlens,
                cu_seqlens_list=ctx.cu_seqlens_list,
                chunk_indices=ctx.chunk_indices,
                chunk_indices_list=ctx.chunk_indices_list,
                chunk_size=ctx.chunk_size,
            )
            if ctx.use_l2norm:
                if l2norm_bwd_mode == "kernel_original":
                    dq = flash_module.l2norm_bwd(q_orig, q_rstd, dq)
                    dk = flash_module.l2norm_bwd(k_orig, k_rstd, dk)
                elif l2norm_bwd_mode == "py_normalized":
                    dq = py_l2norm_bwd_normalized(q_work, q_rstd, dq)
                    dk = py_l2norm_bwd_normalized(k_work, k_rstd, dk)
                elif l2norm_bwd_mode == "py_original":
                    dq = py_l2norm_bwd_original(q_orig, q_rstd, dq)
                    dk = py_l2norm_bwd_original(k_orig, k_rstd, dk)
                else:
                    raise RuntimeError(f"unknown l2norm_bwd_mode={l2norm_bwd_mode!r}")
            return (
                dq.to(q_orig),
                dk.to(k_orig),
                dv.to(v_in),
                dg.to(g_out),
                db.to(beta_in),
                None,
                dh0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    cu_meta, cu_list, chunk_indices, chunk_indices_list = ensure_metadata(
        flash_module, g_ac, cu, case.chunk_size
    )
    scale = case.key_dim ** -0.5
    o_ac, _ = SavedXL2NormFunction.apply(
        q_ac,
        k_ac,
        v_ac,
        g_ac,
        beta_ac,
        float(scale),
        None,
        False,
        cu_meta,
        cu_list,
        chunk_indices,
        chunk_indices_list,
        use_qk_l2norm,
        case.chunk_size,
    )
    o_ac.backward(do)
    torch.npu.synchronize()
    return {
        "output": o_ac,
        "grad_q": q_ac.grad.transpose(1, 2).contiguous(),
        "grad_k": k_ac.grad.transpose(1, 2).contiguous(),
        "grad_v": v_ac.grad.transpose(1, 2).contiguous(),
        "grad_beta": beta_ac.grad,
        "grad_g": g_ac.grad,
        "components": {},
    }


def run_wrapper_ascendc_saved_x_variant(case: cmp.Case, args: argparse.Namespace, flash_module, inputs: tuple[Any, ...], do):
    return run_wrapper_l2norm_bwd_variant(case, args, flash_module, inputs, do, "kernel_original")


def run_wrapper_ascendc_py_l2norm_norm_variant(case: cmp.Case, args: argparse.Namespace, flash_module, inputs: tuple[Any, ...], do):
    return run_wrapper_l2norm_bwd_variant(case, args, flash_module, inputs, do, "py_normalized")


def run_wrapper_ascendc_py_l2norm_orig_variant(case: cmp.Case, args: argparse.Namespace, flash_module, inputs: tuple[Any, ...], do):
    return run_wrapper_l2norm_bwd_variant(case, args, flash_module, inputs, do, "py_original")


def run_variant(
    name: str,
    case: cmp.Case,
    args: argparse.Namespace,
    flash_module,
    triton_dvlocal,
    triton_dhu,
    triton_dqkwg,
    triton_wy,
    inputs: tuple[Any, ...],
    do,
):
    torch = cmp.torch
    q, k, v, beta, g, cu = inputs
    replace_dv_local, replace_dhu, replace_dqkwg, replace_wy = VARIANTS[name]
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
        triton_dvlocal,
        triton_dhu,
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
        replace_dv_local=replace_dv_local,
        replace_dhu=replace_dhu,
        replace_dqkwg=replace_dqkwg,
        replace_wy=replace_wy,
    )

    dq = parts["dq"]
    dk = parts["dk"]
    if use_qk_l2norm:
        q_l2norm_arg = q_hf if args.manual_l2norm_bwd_input == "original" else q_work
        k_l2norm_arg = k_hf if args.manual_l2norm_bwd_input == "original" else k_work
        dq = flash_module.l2norm_bwd(q_l2norm_arg, q_rstd, dq)
        dk = flash_module.l2norm_bwd(k_l2norm_arg, k_rstd, dk)

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
            "dv_local": hf_to_tf(parts["dv_local"]),
            "dv_mid": hf_to_tf(parts["dv_mid"]),
            "dh": parts["dh"].transpose(1, 2).contiguous(),
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
    for name in ("dk_from_dqkwg", "dk_from_wy", "dv_local", "dv_mid"):
        lhs = maybe_unflatten(cu, left["components"][name])
        rhs = maybe_unflatten(cu, right["components"][name])
        out[name] = cmp.tensor_stats(lhs, rhs, args.atol, args.rtol, valid_mask)
    out["dh"] = cmp.tensor_stats(
        left["components"]["dh"],
        right["components"]["dh"],
        args.atol,
        args.rtol,
    )
    return out



def valid_token_mask(case: cmp.Case, cu, device):
    torch = cmp.torch
    if cu is None:
        return torch.ones(case.batch, case.seq_len, device=device, dtype=torch.bool)
    return cmp.varlen_valid_mask(cu, device)


def mask_packed_range(case: cmp.Case, cu, device, start: int, end: int):
    torch = cmp.torch
    if end <= start:
        return None
    if cu is None:
        mask = torch.zeros(case.batch, case.seq_len, device=device, dtype=torch.bool)
        start = max(0, min(start, case.seq_len))
        end = max(start, min(end, case.seq_len))
        mask[:, start:end] = True
        return mask

    cu_list = [int(x) for x in cu.detach().cpu().tolist()]
    batch = len(cu_list) - 1
    max_len = max(cu_list[i + 1] - cu_list[i] for i in range(batch))
    mask = torch.zeros(batch, max_len, device=device, dtype=torch.bool)
    for seq_idx in range(batch):
        seq_start, seq_end = cu_list[seq_idx], cu_list[seq_idx + 1]
        left = max(start, seq_start)
        right = min(end, seq_end)
        if right > left:
            mask[seq_idx, left - seq_start : right - seq_start] = True
    return mask


def sequence_partial_tail_mask(cu, device, chunk_size: int):
    torch = cmp.torch
    if cu is None:
        return None
    cu_list = [int(x) for x in cu.detach().cpu().tolist()]
    batch = len(cu_list) - 1
    max_len = max(cu_list[i + 1] - cu_list[i] for i in range(batch))
    mask = torch.zeros(batch, max_len, device=device, dtype=torch.bool)
    for seq_idx in range(batch):
        length = cu_list[seq_idx + 1] - cu_list[seq_idx]
        tail = length % chunk_size
        if tail:
            mask[seq_idx, length - tail : length] = True
    return mask if bool(mask.any().item()) else None


def packed_index_for_token(cu, seq_idx: int, token_idx: int, seq_len: int) -> int:
    if cu is None:
        return seq_idx * seq_len + token_idx
    cu_list = [int(x) for x in cu.detach().cpu().tolist()]
    return cu_list[seq_idx] + token_idx


def topk_errors(actual, expected, mask, cu, case: cmp.Case, limit: int):
    torch = cmp.torch
    if limit <= 0 or mask is None or not bool(mask.any().item()):
        return []
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    expanded_mask = mask.to(device=actual_f.device, dtype=torch.bool)
    while expanded_mask.ndim < actual_f.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.expand_as(actual_f)
    diff = (actual_f - expected_f).abs()
    selected_count = int(expanded_mask.sum().item())
    if selected_count == 0:
        return []
    scores = diff.masked_fill(~expanded_mask, -1).flatten()
    k = min(int(limit), selected_count)
    values, flat_indices = torch.topk(scores, k=k)
    rows = []
    shape = actual_f.shape
    for value, flat_index in zip(values.detach().cpu().tolist(), flat_indices.detach().cpu().tolist()):
        if value < 0:
            continue
        coords = []
        remainder = int(flat_index)
        for dim in reversed(shape):
            coords.append(remainder % dim)
            remainder //= dim
        coords = list(reversed(coords))
        seq_idx, token_idx = coords[0], coords[1]
        head_idx = coords[2] if len(coords) > 2 else None
        dim_idx = coords[3] if len(coords) > 3 else None
        actual_value = actual_f[tuple(coords)].item()
        expected_value = expected_f[tuple(coords)].item()
        rows.append(
            {
                "seq": int(seq_idx),
                "token": int(token_idx),
                "packed_token": int(packed_index_for_token(cu, seq_idx, token_idx, case.seq_len)),
                "head": None if head_idx is None else int(head_idx),
                "dim": None if dim_idx is None else int(dim_idx),
                "actual": float(actual_value),
                "expected": float(expected_value),
                "abs_diff": float(value),
                "rel_diff": float(value / max(abs(expected_value), 1e-12)),
            }
        )
    return rows


def tail_reports(raw: dict[str, Any], reference: dict[str, Any], case: cmp.Case, cu, args: argparse.Namespace):
    device = raw["grad_k"].device
    actual = maybe_unflatten(cu, raw["grad_k"])
    expected = reference["grad_k"]
    valid = valid_token_mask(case, cu, device)
    reports = {}

    total_tail = case.seq_len % case.chunk_size
    if total_tail:
        packed_tail = mask_packed_range(case, cu, device, case.seq_len - total_tail, case.seq_len)
        if packed_tail is not None and bool(packed_tail.any().item()):
            reports["packed_final_tail"] = {
                "token_count": int(packed_tail.sum().item()),
                "packed_range": [case.seq_len - total_tail, case.seq_len],
                "stats": cmp.tensor_stats(actual, expected, args.atol, args.rtol, packed_tail),
                "top_errors": topk_errors(actual, expected, packed_tail, cu, case, args.tail_topk),
            }
            non_tail = valid & ~packed_tail
            if bool(non_tail.any().item()):
                reports["packed_non_tail"] = {
                    "token_count": int(non_tail.sum().item()),
                    "stats": cmp.tensor_stats(actual, expected, args.atol, args.rtol, non_tail),
                }

    seq_tail = sequence_partial_tail_mask(cu, device, case.chunk_size)
    if seq_tail is not None:
        reports["sequence_partial_tails"] = {
            "token_count": int(seq_tail.sum().item()),
            "stats": cmp.tensor_stats(actual, expected, args.atol, args.rtol, seq_tail),
            "top_errors": topk_errors(actual, expected, seq_tail, cu, case, args.tail_topk),
        }
    return reports


def print_tail_summary(name: str, reports: dict[str, Any]):
    packed_tail = reports.get("packed_final_tail")
    if packed_tail is not None:
        stats = packed_tail["stats"]
        print(
            "    packed_tail_grad_k",
            f"tokens={packed_tail['token_count']}",
            f"range={packed_tail['packed_range']}",
            f"allclose={stats['allclose']}",
            f"max_abs={stats['max_abs']:.6g}",
            f"rms={stats['rms']:.6g}",
            f"mismatch={stats['mismatch_ratio']:.6g}",
            flush=True,
        )
    non_tail = reports.get("packed_non_tail")
    if non_tail is not None:
        stats = non_tail["stats"]
        print(
            "    non_tail_grad_k",
            f"tokens={non_tail['token_count']}",
            f"allclose={stats['allclose']}",
            f"max_abs={stats['max_abs']:.6g}",
            f"rms={stats['rms']:.6g}",
            f"mismatch={stats['mismatch_ratio']:.6g}",
            flush=True,
        )


def print_tail_top_errors(name: str, reports: dict[str, Any]):
    packed_tail = reports.get("packed_final_tail")
    if not packed_tail or not packed_tail.get("top_errors"):
        return
    print(f"\ntop grad_k errors in packed final tail ({name})")
    for row in packed_tail["top_errors"]:
        print(
            "  "
            f"packed={row['packed_token']} seq={row['seq']} tok={row['token']} "
            f"h={row['head']} d={row['dim']} "
            f"actual={row['actual']:.8g} expected={row['expected']:.8g} "
            f"abs={row['abs_diff']:.8g} rel={row['rel_diff']:.8g}"
        )

def failed_tensors(comparisons: dict[str, Any]) -> list[str]:
    return [name for name, stats in comparisons.items() if not stats["allclose"]]


def compact_error(exc: Exception, max_lines: int = 12) -> str:
    lines = str(exc).strip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    keep_head = max(1, max_lines // 3)
    keep_tail = max_lines - keep_head - 1
    return "\n".join([*lines[:keep_head], "...", *lines[-keep_tail:]])


def grad_k_allclose(variant_results: dict[str, Any], name: str) -> bool:
    result = variant_results.get(name)
    if not result or "comparisons" not in result:
        return False
    return bool(result["comparisons"].get("grad_k", {}).get("allclose", False))


def infer_culprit(variant_results: dict[str, Any]) -> str:
    present = set(variant_results)
    if "ascendc" not in present:
        return "Run the ascendc variant to infer a culprit."
    if "comparisons" not in variant_results["ascendc"]:
        return "ascendc variant did not complete; cannot infer a culprit."
    if grad_k_allclose(variant_results, "ascendc"):
        return "wrapper ascendc grad_k already passes; no failing branch in this case."

    if grad_k_allclose(variant_results, "manual_ascendc"):
        return (
            "Real wrapper AscendC fails, but the hand-reconstructed internal AscendC chain passes. "
            "Do not blame dqkwg/WY from this run; first inspect wrapper/autograd parity, saved tensors, "
            "metadata conversion, layout, and l2norm handling."
        )

    if grad_k_allclose(variant_results, "ascendc_py_l2norm_norm"):
        return "Replacing only final l2norm_bwd with PyTorch formula using normalized q/k fixes grad_k; culprit is the l2norm_bwd kernel/implementation, normalized-input contract."
    if grad_k_allclose(variant_results, "ascendc_py_l2norm_orig"):
        return "Replacing only final l2norm_bwd with PyTorch formula using original q/k fixes grad_k; culprit is the l2norm_bwd kernel/implementation, original-input contract."
    if grad_k_allclose(variant_results, "ascendc_saved_x"):
        return "Saving original q/k for existing l2norm_bwd fixes grad_k; culprit is q/k l2norm backward saved-input contract, not a GDN sub-operator."

    full_triton_ok = grad_k_allclose(variant_results, "triton_full")
    if full_triton_ok:
        full_prefix = "Full Triton baseline passes on this shape; AscendC-specific issue confirmed. "
    elif "triton_full" in variant_results:
        full_prefix = "Full Triton baseline did not pass or did not complete; do not use Triton as a replacement yet. "
    else:
        full_prefix = ""

    dvlocal_ok = grad_k_allclose(variant_results, "triton_dvlocal")
    dhu_ok = grad_k_allclose(variant_results, "triton_dhu")
    dvlocal_dhu_ok = grad_k_allclose(variant_results, "triton_dvlocal_dhu")
    dqkwg_ok = grad_k_allclose(variant_results, "triton_dqkwg")
    wy_ok = grad_k_allclose(variant_results, "triton_wy")
    late_both_ok = grad_k_allclose(variant_results, "triton_both")
    all_ok = grad_k_allclose(variant_results, "triton_all")
    if dvlocal_ok:
        return full_prefix + "Most likely culprit: npu_chunk_bwd_dv_local."
    if dhu_ok:
        return full_prefix + "Most likely culprit: npu_chunk_gated_delta_rule_bwd_dhu."
    if dvlocal_dhu_ok:
        return full_prefix + "Replacing dv_local+dhu fixes grad_k; inspect both upstream backward ops."
    if dqkwg_ok and not wy_ok:
        return full_prefix + "Most likely culprit: npu_chunk_bwd_dqkwg."
    if wy_ok and not dqkwg_ok:
        return full_prefix + "Most likely culprit: npu_prepare_wy_repr_bwd_da/full."
    if dqkwg_ok and wy_ok:
        return full_prefix + "Both late-branch replacements make grad_k pass; inspect component deltas."
    if late_both_ok:
        return full_prefix + "Only replacing dqkwg+WY makes grad_k pass; error is split across late branches."
    if all_ok:
        return full_prefix + "Only replacing dv_local+dhu+dqkwg+WY makes grad_k pass; upstream and late branches both contribute or Triton layouts differ."
    return full_prefix + "No validated Triton replacement fixed grad_k; check controls before blaming a single op."


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

        full_triton_gdn = load_full_triton_gdn()
        triton_dvlocal, triton_dhu, triton_dqkwg, triton_wy = load_local_triton()
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

        if args.variant == "all":
            names = [
                name for name, flags in VARIANTS.items()
                if args.include_dhu_hybrid or not flags[1]
            ]
        else:
            names = [args.variant]
        raw_results = {}
        variant_results = {}
        for name in names:
            print(f"==> {name}", flush=True)
            try:
                if name == "ascendc":
                    raw = run_wrapper_ascendc_variant(case, args, flash_module, inputs, do)
                elif name == "ascendc_saved_x":
                    raw = run_wrapper_ascendc_saved_x_variant(case, args, flash_module, inputs, do)
                elif name == "ascendc_py_l2norm_norm":
                    raw = run_wrapper_ascendc_py_l2norm_norm_variant(case, args, flash_module, inputs, do)
                elif name == "ascendc_py_l2norm_orig":
                    raw = run_wrapper_ascendc_py_l2norm_orig_variant(case, args, flash_module, inputs, do)
                elif name == "triton_full":
                    raw = run_full_triton_variant(case, args, full_triton_gdn, inputs, do)
                else:
                    raw = run_variant(
                        name,
                        case,
                        args,
                        flash_module,
                        triton_dvlocal,
                        triton_dhu,
                        triton_dqkwg,
                        triton_wy,
                        inputs,
                        do,
                    )
                comparisons = compare_outputs(raw, reference, inputs[-1], args)
                tails = tail_reports(raw, reference, case, inputs[-1], args)
            except Exception as exc:  # pylint: disable=broad-except
                error_text = compact_error(exc)
                variant_results[name] = {
                    "kind": "wrapper_ascendc" if name == "ascendc" else ("full_triton" if name == "triton_full" else "hybrid_or_manual_ascendc"),
                    "replace_dv_local": VARIANTS[name][0],
                    "replace_dhu": VARIANTS[name][1],
                    "replace_dqkwg": VARIANTS[name][2],
                    "replace_wy": VARIANTS[name][3],
                    "passed": False,
                    "failed_tensors": ["__variant_error__"],
                    "error": error_text,
                }
                print("    ERROR", error_text.replace("\n", " | "), flush=True)
                continue

            raw_results[name] = raw
            variant_results[name] = {
                "kind": "wrapper_ascendc" if name == "ascendc" else ("full_triton" if name == "triton_full" else "hybrid_or_manual_ascendc"),
                "replace_dv_local": VARIANTS[name][0],
                "replace_dhu": VARIANTS[name][1],
                "replace_dqkwg": VARIANTS[name][2],
                "replace_wy": VARIANTS[name][3],
                "passed": all(stats["allclose"] for stats in comparisons.values()),
                "failed_tensors": failed_tensors(comparisons),
                "comparisons": comparisons,
                "tail_reports": tails,
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
            print_tail_summary(name, tails)

        if "ascendc" in variant_results:
            print_tail_top_errors("ascendc", variant_results["ascendc"].get("tail_reports", {}))
        elif len(variant_results) == 1:
            only_name = next(iter(variant_results))
            print_tail_top_errors(only_name, variant_results[only_name].get("tail_reports", {}))

        component_deltas = {}
        if "manual_ascendc" in raw_results:
            for right_name in (
                "triton_dvlocal",
                "triton_dqkwg",
                "triton_wy",
                "triton_both",
                "triton_dvlocal_dhu",
                "triton_all",
            ):
                if right_name in raw_results:
                    component_deltas[f"manual_ascendc_vs_{right_name}"] = component_delta(
                        raw_results["manual_ascendc"], raw_results[right_name], inputs[-1], args
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
            if "error" in result:
                print(
                    f"{name:14s} passed=False error=True "
                    f"failed={','.join(result['failed_tensors'])}"
                )
                continue
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
            print("\ncomponent deltas")
            for delta_name, delta in component_deltas.items():
                print(delta_name)
                for name, stats in delta.items():
                    print(
                        f"  {name:14s} max_abs={stats['max_abs']:.6g} "
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
