#!/usr/bin/env python3
"""Check the final q/k l2norm_bwd call in the real GDN backward context.

This runs the FLA-NPU GDN forward/backward core to obtain the actual upstream
core dq/dk tensors, then compares the real l2norm_bwd call against the PyTorch
formula on exactly those tensors.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import compare_gdn_precision as cmp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check real l2norm_bwd in actual GDN dq/dk context.")
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
    parser.add_argument("--output-json", type=str)
    parser.add_argument("--topk", type=int, default=16)
    return parser.parse_args()


def load_flash_module(fla_repo: Path):
    if str(fla_repo) not in sys.path:
        sys.path.insert(0, str(fla_repo))
    example_path = fla_repo / "examples" / "flash_gated_delta_rule.py"
    spec = importlib.util.spec_from_file_location("fla_npu_l2norm_context_check", example_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {example_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def broadcast_rstd(rstd, target):
    if rstd.ndim == target.ndim - 1:
        return rstd.unsqueeze(-1)
    return rstd


def py_l2norm_bwd_normalized(y, rstd, dy):
    rstd = broadcast_rstd(rstd, dy)
    dot = (dy * y).sum(dim=-1, keepdim=True)
    return ((dy - y * dot) * rstd).to(y.dtype)


def py_l2norm_bwd_original(x, rstd, dy):
    rstd = broadcast_rstd(rstd, dy)
    y = (x * rstd).to(x.dtype)
    dot = (dy * y).sum(dim=-1, keepdim=True)
    return ((dy - y * dot) * rstd).to(x.dtype)


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


def run_reference(case: cmp.Case, args: argparse.Namespace, inputs: tuple[Any, ...], do):
    torch = cmp.torch
    q, k, v, beta, g, cu = inputs
    q_ref, k_ref, v_ref, beta_ref, g_ref = [cmp.clone_leaf(x) for x in (q, k, v, beta, g)]

    if cu is not None:
        q_in, k_in, v_in, beta_in, g_in = cmp.varlen_to_nonvarlen(cu, q_ref, k_ref, v_ref, beta_ref, g_ref)
        for tensor in (q_in, k_in, v_in, beta_in, g_in):
            tensor.retain_grad()
    else:
        q_in, k_in, v_in, beta_in, g_in = q_ref, k_ref, v_ref, beta_ref, g_ref

    o_ref, _ = cmp.ref_torch_chunk_gated_delta_rule(
        q_in,
        k_in,
        v_in,
        g_in,
        beta_in,
        chunk_size=case.chunk_size,
        use_qk_l2norm_in_kernel=True,
    )
    do_ref = cmp.varlen_to_nonvarlen(cu, do) if cu is not None else do
    o_ref.backward(do_ref)
    torch.npu.synchronize()
    return {"grad_q": q_in.grad, "grad_k": k_in.grad}


def segment_lengths(case: cmp.Case) -> list[int]:
    if case.cu_seqlens is None:
        return [case.seq_len]
    return [b - a for a, b in zip(case.cu_seqlens, case.cu_seqlens[1:])]


def tail_mask_for_bhtd(case: cmp.Case, device):
    torch = cmp.torch
    lengths = segment_lengths(case)
    tails = [length % case.chunk_size for length in lengths]
    if not any(tails):
        return None
    mask = torch.zeros(case.batch, case.heads, case.seq_len, device=device, dtype=torch.bool)
    if case.cu_seqlens is None:
        for length, rem in zip(lengths, tails):
            if rem:
                mask[:, :, length - rem : length] = True
    else:
        for start, length, rem in zip(case.cu_seqlens[:-1], lengths, tails):
            if rem:
                mask[:, :, start + length - rem : start + length] = True
    return mask


def segment_info_for_packed_token(case: cmp.Case, token: int):
    if case.cu_seqlens is None:
        length = case.seq_len
        rem = length % case.chunk_size
        return {
            "seq": 0,
            "token_in_seq": token,
            "seq_len": length,
            "tail_len": rem,
            "in_tail": bool(rem and token >= length - rem),
        }
    for seq, (start, end) in enumerate(zip(case.cu_seqlens, case.cu_seqlens[1:])):
        if start <= token < end:
            length = end - start
            rem = length % case.chunk_size
            token_in_seq = token - start
            return {
                "seq": seq,
                "token_in_seq": token_in_seq,
                "seq_len": length,
                "tail_len": rem,
                "in_tail": bool(rem and token_in_seq >= length - rem),
            }
    return {"seq": None, "token_in_seq": None, "seq_len": None, "tail_len": None, "in_tail": False}


def tail_mask_for_bthd(case: cmp.Case, device):
    torch = cmp.torch
    lengths = segment_lengths(case)
    tails = [length % case.chunk_size for length in lengths]
    if not any(tails):
        return None
    max_len = max(lengths)
    mask = torch.zeros(len(lengths) if case.cu_seqlens is not None else case.batch, max_len, case.heads, device=device, dtype=torch.bool)
    for seq_idx, (length, rem) in enumerate(zip(lengths, tails)):
        if rem == 0:
            continue
        tail_start = length - rem
        tail_end = length
        if case.cu_seqlens is not None:
            mask[seq_idx, tail_start:tail_end, :] = True
        else:
            mask[:, tail_start:tail_end, :] = True
    return mask


def to_bthd_for_compare(cu, tensor_bhtd):
    tensor_bthd = tensor_bhtd.transpose(1, 2).contiguous()
    return cmp.varlen_to_nonvarlen(cu, tensor_bthd) if cu is not None else tensor_bthd


def compare(actual, expected, args: argparse.Namespace, mask=None):
    return cmp.tensor_stats(actual, expected, args.atol, args.rtol, mask)


def tensor_meta(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "is_contiguous": bool(tensor.is_contiguous()),
        "dtype": str(tensor.dtype),
        "storage_offset": int(tensor.storage_offset()),
    }


def tensor_value_stats(tensor, mask=None):
    torch = cmp.torch
    t = tensor.detach().float()
    if mask is not None:
        mask = mask.to(device=t.device, dtype=torch.bool)
        while mask.ndim < t.ndim:
            mask = mask.unsqueeze(-1)
        t = t[mask.expand_as(t)]
    return {
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "rms": float((t * t).mean().sqrt().item()),
        "zero_ratio": float((t == 0).float().mean().item()),
        "finite": bool(t.isfinite().all().item()),
        "numel": int(t.numel()),
    }


def compare_kernel_on_dy(label, flash_module, x_norm, rstd, dy, args: argparse.Namespace, case: cmp.Case, tail_mask_bhtd=None):
    kernel = flash_module.l2norm_bwd(x_norm, rstd, dy)
    py = py_l2norm_bwd_normalized(x_norm, rstd, dy)
    stats = compare(kernel, py, args)
    tail_stats = compare(kernel, py, args, tail_mask_bhtd) if tail_mask_bhtd is not None else None
    return {
        "label": label,
        "stats": stats,
        "tail_stats": tail_stats,
        "dy_stats": tensor_value_stats(dy),
        "tail_dy_stats": tensor_value_stats(dy, tail_mask_bhtd) if tail_mask_bhtd is not None else None,
        "top1": topk_error_details(kernel, py, x_norm, rstd, dy, case, args, None)[:1],
        "tail_top1": topk_error_details(kernel, py, x_norm, rstd, dy, case, args, tail_mask_bhtd)[:1]
        if tail_mask_bhtd is not None
        else [],
    }


def expanded_mask(mask, tensor):
    torch = cmp.torch
    if mask is None:
        return None
    mask = mask.to(device=tensor.device, dtype=torch.bool)
    while mask.ndim < tensor.ndim:
        mask = mask.unsqueeze(-1)
    return mask.expand_as(tensor)


def make_dy_controls(dy, case: cmp.Case, args: argparse.Namespace, tail_mask_bhtd=None):
    torch = cmp.torch
    controls = {"real": dy}

    torch.manual_seed(args.seed + 101)
    rand = torch.randn_like(dy)
    dy_rms = dy.detach().float().square().mean().sqrt().clamp_min(1e-12)
    rand_rms = rand.detach().float().square().mean().sqrt().clamp_min(1e-12)
    controls["random_same_rms"] = (rand * (dy_rms / rand_rms)).to(dy.dtype)

    torch.manual_seed(args.seed + 102)
    flat = dy.flatten()
    perm = torch.randperm(flat.numel(), device=dy.device)
    controls["shuffle_flat"] = flat[perm].reshape_as(dy).contiguous()

    torch.manual_seed(args.seed + 103)
    token_perm = torch.randperm(dy.shape[2], device=dy.device)
    controls["shuffle_tokens"] = dy[:, :, token_perm, :].contiguous()

    mask = expanded_mask(tail_mask_bhtd, dy)
    if mask is not None:
        zero = torch.zeros_like(dy)
        controls["tail_only"] = torch.where(mask, dy, zero)
        controls["tail_zero"] = torch.where(mask, zero, dy)

        torch.manual_seed(args.seed + 104)
        rand_tail = torch.randn_like(dy)
        tail_values = dy.detach().float()[mask]
        tail_rms = tail_values.square().mean().sqrt().clamp_min(1e-12)
        rand_tail_rms = rand_tail.detach().float()[mask].square().mean().sqrt().clamp_min(1e-12)
        controls["random_tail_same_rms"] = torch.where(mask, (rand_tail * (tail_rms / rand_tail_rms)).to(dy.dtype), zero)

    return controls


def topk_error_details(actual, expected, x_norm, rstd, dy, case: cmp.Case, args: argparse.Namespace, mask=None):
    torch = cmp.torch
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    diff = (actual_f - expected_f).abs()
    if mask is not None:
        mask = mask.to(device=diff.device, dtype=torch.bool)
        while mask.ndim < diff.ndim:
            mask = mask.unsqueeze(-1)
        diff = diff.masked_fill(~mask.expand_as(diff), -1)
    flat = diff.flatten()
    k = min(args.topk, flat.numel())
    vals, idxs = torch.topk(flat, k)
    shape = list(diff.shape)
    rows = []
    for value, flat_idx in zip(vals.detach().cpu().tolist(), idxs.detach().cpu().tolist()):
        if value < 0:
            continue
        idx = int(flat_idx)
        dim = idx % shape[3]
        idx //= shape[3]
        token = idx % shape[2]
        idx //= shape[2]
        head = idx % shape[1]
        batch = idx // shape[1]
        rows.append(
            {
                "abs_diff": float(value),
                "kernel": float(actual_f[batch, head, token, dim].item()),
                "py": float(expected_f[batch, head, token, dim].item()),
                "dk_core": float(dy.detach().float()[batch, head, token, dim].item()),
                "x_norm": float(x_norm.detach().float()[batch, head, token, dim].item()),
                "rstd": float(rstd.detach().float()[batch, head, token].item()),
                "batch": int(batch),
                "head": int(head),
                "packed_token": int(token),
                "dim": int(dim),
                **segment_info_for_packed_token(case, int(token)),
            }
        )
    return rows


def run_case(case: cmp.Case, args: argparse.Namespace, flash_module) -> dict[str, Any]:
    torch = cmp.torch
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(args.device)
    if hasattr(torch.npu, "set_compile_mode"):
        torch.npu.set_compile_mode(jit_compile=False)

    inputs = cmp.make_inputs(case, device, dtype, args.seed)
    q, k, v, beta, g, cu = inputs
    q_bhtd, k_bhtd, v_bhtd = [x.transpose(1, 2).contiguous() for x in (q, k, v)]

    torch.manual_seed(args.seed + 1)
    do = torch.randn(case.batch, case.seq_len, case.heads, case.value_dim, dtype=dtype, device=device)
    reference = run_reference(case, args, inputs, do)

    q_norm, q_rstd = flash_module.l2norm_fwd(q_bhtd)
    k_norm, k_rstd = flash_module.l2norm_fwd(k_bhtd)
    cu_meta, cu_list, chunk_indices, chunk_indices_list = ensure_metadata(flash_module, g, cu, case.chunk_size)
    scale = case.key_dim ** -0.5

    g_out, _output, A, _final_state = flash_module.flash_chunk_gated_delta_rule_fwd(
        q=q_norm,
        k=k_norm,
        v=v_bhtd,
        g=g,
        beta=beta,
        scale=float(scale),
        initial_state=None,
        output_final_state=False,
        cu_seqlens=cu_meta,
        cu_seqlens_list=cu_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        chunk_size=case.chunk_size,
    )

    dq_core, dk_core, _dv, _db, _dg, _dh0 = flash_module.flash_chunk_gated_delta_rule_bwd(
        q=q_norm,
        k=k_norm,
        v=v_bhtd,
        g=g_out,
        beta=beta,
        A=A,
        scale=float(scale),
        initial_state=None,
        do=do,
        dht=None,
        cu_seqlens=cu_meta,
        cu_seqlens_list=cu_list,
        chunk_indices=chunk_indices,
        chunk_indices_list=chunk_indices_list,
        chunk_size=case.chunk_size,
    )

    dq_kernel_norm = flash_module.l2norm_bwd(q_norm, q_rstd, dq_core)
    dk_kernel_norm = flash_module.l2norm_bwd(k_norm, k_rstd, dk_core)
    dk_kernel_dy_contig = flash_module.l2norm_bwd(k_norm, k_rstd, dk_core.contiguous())
    dk_kernel_dy_clone = flash_module.l2norm_bwd(k_norm, k_rstd, dk_core.detach().clone())
    dk_kernel_all_clone = flash_module.l2norm_bwd(
        k_norm.detach().clone().contiguous(),
        k_rstd.detach().clone().contiguous(),
        dk_core.detach().clone().contiguous(),
    )
    dq_py_norm = py_l2norm_bwd_normalized(q_norm, q_rstd, dq_core)
    dk_py_norm = py_l2norm_bwd_normalized(k_norm, k_rstd, dk_core)
    dq_py_orig = py_l2norm_bwd_original(q_bhtd, q_rstd, dq_core)
    dk_py_orig = py_l2norm_bwd_original(k_bhtd, k_rstd, dk_core)
    torch.npu.synchronize()

    dq_kernel_bthd = to_bthd_for_compare(cu, dq_kernel_norm)
    dk_kernel_bthd = to_bthd_for_compare(cu, dk_kernel_norm)
    dk_kernel_dy_contig_bthd = to_bthd_for_compare(cu, dk_kernel_dy_contig)
    dk_kernel_dy_clone_bthd = to_bthd_for_compare(cu, dk_kernel_dy_clone)
    dk_kernel_all_clone_bthd = to_bthd_for_compare(cu, dk_kernel_all_clone)
    dq_py_norm_bthd = to_bthd_for_compare(cu, dq_py_norm)
    dk_py_norm_bthd = to_bthd_for_compare(cu, dk_py_norm)
    dq_py_orig_bthd = to_bthd_for_compare(cu, dq_py_orig)
    dk_py_orig_bthd = to_bthd_for_compare(cu, dk_py_orig)
    tail_mask = tail_mask_for_bthd(case, device)
    tail_mask_bhtd = tail_mask_for_bhtd(case, device)

    dy_control_reports = {
        name: compare_kernel_on_dy(name, flash_module, k_norm, k_rstd, dy_value, args, case, tail_mask_bhtd)
        for name, dy_value in make_dy_controls(dk_core, case, args, tail_mask_bhtd).items()
    }

    comparisons = {
        "q_kernel_vs_py_norm": compare(dq_kernel_bthd, dq_py_norm_bthd, args),
        "k_kernel_vs_py_norm": compare(dk_kernel_bthd, dk_py_norm_bthd, args),
        "k_kernel_dy_contig_vs_py_norm": compare(dk_kernel_dy_contig_bthd, dk_py_norm_bthd, args),
        "k_kernel_dy_clone_vs_py_norm": compare(dk_kernel_dy_clone_bthd, dk_py_norm_bthd, args),
        "k_kernel_all_clone_vs_py_norm": compare(dk_kernel_all_clone_bthd, dk_py_norm_bthd, args),
        "q_kernel_vs_ref": compare(dq_kernel_bthd, reference["grad_q"], args),
        "k_kernel_vs_ref": compare(dk_kernel_bthd, reference["grad_k"], args),
        "q_py_norm_vs_ref": compare(dq_py_norm_bthd, reference["grad_q"], args),
        "k_py_norm_vs_ref": compare(dk_py_norm_bthd, reference["grad_k"], args),
        "q_py_orig_vs_ref": compare(dq_py_orig_bthd, reference["grad_q"], args),
        "k_py_orig_vs_ref": compare(dk_py_orig_bthd, reference["grad_k"], args),
    }
    tail_reports = {}
    if tail_mask is not None:
        for key, actual, expected in (
            ("q_kernel_vs_py_norm", dq_kernel_bthd, dq_py_norm_bthd),
            ("k_kernel_vs_py_norm", dk_kernel_bthd, dk_py_norm_bthd),
            ("k_kernel_dy_contig_vs_py_norm", dk_kernel_dy_contig_bthd, dk_py_norm_bthd),
            ("k_kernel_dy_clone_vs_py_norm", dk_kernel_dy_clone_bthd, dk_py_norm_bthd),
            ("k_kernel_all_clone_vs_py_norm", dk_kernel_all_clone_bthd, dk_py_norm_bthd),
            ("q_kernel_vs_ref", dq_kernel_bthd, reference["grad_q"]),
            ("k_kernel_vs_ref", dk_kernel_bthd, reference["grad_k"]),
            ("q_py_norm_vs_ref", dq_py_norm_bthd, reference["grad_q"]),
            ("k_py_norm_vs_ref", dk_py_norm_bthd, reference["grad_k"]),
        ):
            tail_reports[key] = compare(actual, expected, args, tail_mask)

    return {
        "case": case.__dict__,
        "shape_bhtd": list(q_bhtd.shape),
        "dtype": args.dtype,
        "device": str(device),
        "seed": args.seed,
        "segment_lengths": segment_lengths(case),
        "segment_tails": [length % case.chunk_size for length in segment_lengths(case)],
        "tensor_meta": {
            "q_norm": tensor_meta(q_norm),
            "k_norm": tensor_meta(k_norm),
            "q_rstd": tensor_meta(q_rstd),
            "k_rstd": tensor_meta(k_rstd),
            "dq_core": tensor_meta(dq_core),
            "dk_core": tensor_meta(dk_core),
        },
        "tensor_value_stats": {
            "dq_core": tensor_value_stats(dq_core),
            "dk_core": tensor_value_stats(dk_core),
        },
        "tensor_tail_value_stats": {
            "dq_core": tensor_value_stats(dq_core, tail_mask_bhtd) if tail_mask_bhtd is not None else None,
            "dk_core": tensor_value_stats(dk_core, tail_mask_bhtd) if tail_mask_bhtd is not None else None,
        },
        "topk_errors": {
            "k_kernel_vs_py_norm": topk_error_details(
                dk_kernel_norm, dk_py_norm, k_norm, k_rstd, dk_core, case, args
            ),
            "tail_k_kernel_vs_py_norm": topk_error_details(
                dk_kernel_norm, dk_py_norm, k_norm, k_rstd, dk_core, case, args, tail_mask_bhtd
            )
            if tail_mask_bhtd is not None
            else [],
        },
        "dy_control_reports": dy_control_reports,
        "passed": (
            comparisons["q_kernel_vs_py_norm"]["allclose"]
            and comparisons["k_kernel_vs_py_norm"]["allclose"]
            and comparisons["q_kernel_vs_ref"]["allclose"]
            and comparisons["k_kernel_vs_ref"]["allclose"]
        ),
        "comparisons": comparisons,
        "tail_reports": tail_reports,
    }


def write_payload(payload: dict[str, Any], output_json: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")


def main() -> int:
    args = parse_args()
    torch, _ = cmp.import_torch_runtime()
    try:
        cmp.check_npu_runtime()
        fla_repo = cmp.find_fla_repo(args.fla_repo)
        flash_module = load_flash_module(fla_repo)
        case = cmp.override_case(cmp.CASES[args.case], args)
        result = run_case(case, args, flash_module)
    except Exception as exc:  # pylint: disable=broad-except
        payload = {"comparison": "real_l2norm_bwd_in_gdn_context", "error": str(exc), "passed": False}
        write_payload(payload, args.output_json)
        return 2

    payload = {
        "comparison": "real_l2norm_bwd_in_gdn_context",
        "fla_repo": str(fla_repo),
        "torch": torch.__version__,
        **result,
    }
    write_payload(payload, args.output_json)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
