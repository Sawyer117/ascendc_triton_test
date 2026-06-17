#!/usr/bin/env python3
"""Compare l2norm_fwd/bwd against Python autograd for GDN q/k-shaped tensors."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any

import compare_gdn_precision as cmp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check l2norm_fwd/bwd precision and backward contract.")
    parser.add_argument("--case", choices=cmp.CASES.keys(), default="varlen")
    parser.add_argument("--creative-repo", default=None, help="Only used for --source shim.")
    parser.add_argument("--source", choices=["external", "shim"], default="external")
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


def install_shim(creative_repo_arg: str | None) -> None:
    import compare_creative_gdn_pair as pair

    creative_repo = pair.find_creative_repo(creative_repo_arg)
    package = pair.install_creative_file_package(creative_repo)
    pair.install_mindspeed_triton_shim(package)


def load_l2norm(source: str, creative_repo_arg: str | None):
    if source == "shim":
        install_shim(creative_repo_arg)
    try:
        module = importlib.import_module("mindspeed.lite.ops.triton.l2norm")
    except ModuleNotFoundError as exc:
        return None, {
            "skipped": True,
            "skip_reason": f"cannot import mindspeed.lite.ops.triton.l2norm: {exc}",
            "source": source,
            "source_is_real_kernel": False,
        }
    return (module.l2norm_fwd, module.l2norm_bwd), None


def python_l2norm(x, eps: float = 1e-6):
    torch = cmp.torch
    return (x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)).to(x.dtype)


def unwrap_bwd(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise RuntimeError(f"l2norm_bwd returned {len(value)} values; expected 1")
        return value[0]
    return value


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


def run_case(case: cmp.Case, args: argparse.Namespace, l2norm_fwd, l2norm_bwd):
    torch = cmp.torch
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(args.device)
    if hasattr(torch.npu, "set_compile_mode"):
        torch.npu.set_compile_mode(jit_compile=False)

    q, _, _, _, _, _ = cmp.make_inputs(case, device, dtype, args.seed)
    x_ref = cmp.clone_leaf(q)
    x_op = q.detach().clone()

    torch.manual_seed(args.seed + 17)
    do = torch.randn_like(x_ref)

    y_ref = python_l2norm(x_ref)
    y_ref.backward(do)
    dx_ref = x_ref.grad

    y_op, rstd = l2norm_fwd(x_op)
    dx_from_original = unwrap_bwd(l2norm_bwd(x_op, rstd, do))
    dx_from_output = unwrap_bwd(l2norm_bwd(y_op, rstd, do))
    torch.npu.synchronize()

    comparisons = {
        "output": cmp.tensor_stats(y_op, y_ref, args.atol, args.rtol),
        "bwd_from_original_input": cmp.tensor_stats(dx_from_original, dx_ref, args.atol, args.rtol),
        "bwd_from_normalized_output": cmp.tensor_stats(dx_from_output, dx_ref, args.atol, args.rtol),
    }
    failed = [name for name, stats in comparisons.items() if not stats["allclose"]]
    return {
        "case": case.__dict__,
        "shape_summary": segment_summary(case),
        "dtype": args.dtype,
        "device": str(device),
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "source": args.source,
        "source_is_real_kernel": args.source == "external",
        "passed": comparisons["output"]["allclose"] and (
            comparisons["bwd_from_original_input"]["allclose"]
            or comparisons["bwd_from_normalized_output"]["allclose"]
        ),
        "failed_tensors": failed,
        "comparisons": comparisons,
        "contract": {
            "original_input_ok": comparisons["bwd_from_original_input"]["allclose"],
            "normalized_output_ok": comparisons["bwd_from_normalized_output"]["allclose"],
            "flash_old_path_uses_normalized_output": True,
        },
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
        payload = {"skipped": True, "skip_reason": f"torch_npu is unavailable: {exc}", "source": args.source}
        write_payload(payload, args.output_json)
        return 3
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        payload = {"skipped": True, "skip_reason": "NPU is unavailable", "source": args.source}
        write_payload(payload, args.output_json)
        return 3

    funcs, skipped = load_l2norm(args.source, args.creative_repo)
    if skipped:
        write_payload(skipped, args.output_json)
        return 3

    case = cmp.override_case(cmp.CASES[args.case], args)
    try:
        result = run_case(case, args, funcs[0], funcs[1])
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    payload = {"comparison": "l2norm_fwd_bwd_vs_python_autograd", "torch": torch.__version__, **result}
    write_payload(payload, args.output_json)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
