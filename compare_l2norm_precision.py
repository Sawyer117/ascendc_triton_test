#!/usr/bin/env python3
"""Self-contained l2norm backward-contract checks for GDN q/k-shaped tensors.

This script deliberately does not import mindspeed. It compares pure Python
formulas against PyTorch autograd to make the l2norm_bwd contract explicit.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import compare_gdn_precision as cmp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-contained l2norm backward-contract precision check.")
    parser.add_argument("--case", choices=cmp.CASES.keys(), default="varlen")
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
    return parser.parse_args()


def python_l2norm(x, eps: float = 1e-6):
    torch = cmp.torch
    rstd = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
    return (x * rstd).to(x.dtype), rstd


def bwd_original_input_contract(x, rstd, dy):
    """Backward formula when the first argument is the original, unnormalized x."""
    y = (x * rstd).to(x.dtype)
    dot = (dy * y).sum(dim=-1, keepdim=True)
    return ((dy - y * dot) * rstd).to(x.dtype)


def bwd_normalized_output_contract(y, rstd, dy):
    """Backward formula when the first argument is already normalized output y."""
    dot = (dy * y).sum(dim=-1, keepdim=True)
    return ((dy - y * dot) * rstd).to(y.dtype)


def segment_summary(case: cmp.Case):
    if case.cu_seqlens is None:
        lengths = [case.seq_len]
    else:
        lengths = [b - a for a, b in zip(case.cu_seqlens, case.cu_seqlens[1:])]
    return {
        "total_tokens": case.seq_len,
        "chunk_size": case.chunk_size,
        "total_tail": case.seq_len % case.chunk_size,
        "segment_lengths": lengths,
        "segment_tails": [length % case.chunk_size for length in lengths],
    }


def run_case(case: cmp.Case, args: argparse.Namespace):
    torch = cmp.torch
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(args.device)
    if hasattr(torch.npu, "set_compile_mode"):
        torch.npu.set_compile_mode(jit_compile=False)

    q, _, _, _, _, _ = cmp.make_inputs(case, device, dtype, args.seed)
    x_ref = cmp.clone_leaf(q)
    x_formula = q.detach().clone()

    torch.manual_seed(args.seed + 17)
    dy = torch.randn_like(x_ref)

    y_ref, _ = python_l2norm(x_ref)
    y_ref.backward(dy)
    dx_ref = x_ref.grad

    y_formula, rstd = python_l2norm(x_formula)
    dx_original_contract = bwd_original_input_contract(x_formula, rstd, dy)
    dx_normalized_contract = bwd_normalized_output_contract(y_formula, rstd, dy)

    # This is the exact failure mode the old flash wrapper can hit if l2norm_bwd
    # expects original x while the wrapper passes saved normalized q/k.
    dx_flash_call_if_original_contract = bwd_original_input_contract(y_formula, rstd, dy)

    # This is the same old flash call if l2norm_bwd is intentionally designed
    # to accept normalized output y.
    dx_flash_call_if_normalized_contract = bwd_normalized_output_contract(y_formula, rstd, dy)
    torch.npu.synchronize()

    comparisons = {
        "output": cmp.tensor_stats(y_formula, y_ref, args.atol, args.rtol),
        "bwd_original_input_contract": cmp.tensor_stats(dx_original_contract, dx_ref, args.atol, args.rtol),
        "bwd_normalized_output_contract": cmp.tensor_stats(dx_normalized_contract, dx_ref, args.atol, args.rtol),
        "flash_old_call_if_bwd_expects_original": cmp.tensor_stats(
            dx_flash_call_if_original_contract, dx_ref, args.atol, args.rtol
        ),
        "flash_old_call_if_bwd_expects_normalized": cmp.tensor_stats(
            dx_flash_call_if_normalized_contract, dx_ref, args.atol, args.rtol
        ),
    }
    contracts = {
        "original_input_contract_ok": comparisons["bwd_original_input_contract"]["allclose"],
        "normalized_output_contract_ok": comparisons["bwd_normalized_output_contract"]["allclose"],
        "old_flash_call_ok_if_bwd_expects_original": comparisons[
            "flash_old_call_if_bwd_expects_original"
        ]["allclose"],
        "old_flash_call_ok_if_bwd_expects_normalized": comparisons[
            "flash_old_call_if_bwd_expects_normalized"
        ]["allclose"],
    }
    return {
        "case": case.__dict__,
        "shape_summary": segment_summary(case),
        "dtype": args.dtype,
        "device": str(device),
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "source": "self_contained_formula",
        "imports_mindspeed": False,
        "passed": comparisons["output"]["allclose"]
        and contracts["original_input_contract_ok"]
        and contracts["normalized_output_contract_ok"],
        "comparisons": comparisons,
        "contracts": contracts,
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
        import torch_npu  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import
    except ImportError as exc:
        payload = {"skipped": True, "skip_reason": f"torch_npu is unavailable: {exc}"}
        write_payload(payload, args.output_json)
        return 3
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        payload = {"skipped": True, "skip_reason": "NPU is unavailable"}
        write_payload(payload, args.output_json)
        return 3

    case = cmp.override_case(cmp.CASES[args.case], args)
    try:
        result = run_case(case, args)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    payload = {"comparison": "self_contained_l2norm_contracts_vs_python_autograd", "torch": torch.__version__, **result}
    write_payload(payload, args.output_json)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
