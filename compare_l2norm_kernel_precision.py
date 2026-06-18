#!/usr/bin/env python3
"""Isolate real l2norm_fwd/l2norm_bwd kernel precision against PyTorch autograd.

Unlike compare_l2norm_precision.py, this script imports the l2norm functions
from the FLA-NPU flash_gated_delta_rule module and calls the real kernel path
directly.  It tests both possible backward contracts:

    l2norm_bwd(normalized_y, rstd, dy)
    l2norm_bwd(original_x, rstd, dy)

The GDN wrapper uses B,H,T,D tensors internally, so BHTD is the default layout.
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
    parser = argparse.ArgumentParser(description="Compare real l2norm kernels with PyTorch autograd.")
    parser.add_argument("--case", choices=cmp.CASES.keys(), default="varlen")
    parser.add_argument(
        "--fla-repo",
        default=os.environ.get("FLA_NPU_REPO"),
        help="Path to flash-linear-attention-npu. Defaults to ./flash-linear-attention-npu if present.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--key-dim", type=int)
    parser.add_argument("--value-dim", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--cu-seqlens", type=str)
    parser.add_argument("--layout", choices=["BHTD", "BTHD"], default="BHTD")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--tail-topk", type=int, default=8)
    parser.add_argument("--output-json", type=str)
    return parser.parse_args()


def find_fla_repo(path_arg: str | None) -> Path:
    candidates: list[Path] = []
    if path_arg:
        candidates.append(Path(path_arg))
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            script_dir / "flash-linear-attention-npu",
            cwd / "flash-linear-attention-npu",
            cwd,
            cwd.parent / "flash-linear-attention-npu",
        ]
    )
    for candidate in candidates:
        example = candidate / "examples" / "flash_gated_delta_rule.py"
        if example.is_file():
            return candidate.resolve()
    raise RuntimeError(
        "Cannot find flash-linear-attention-npu. Pass --fla-repo /path/to/flash-linear-attention-npu."
    )


def load_l2norm_kernels(fla_repo: Path):
    if str(fla_repo) not in sys.path:
        sys.path.insert(0, str(fla_repo))
    example_path = fla_repo / "examples" / "flash_gated_delta_rule.py"
    spec = importlib.util.spec_from_file_location("fla_npu_l2norm_kernel_check", example_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {example_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [name for name in ("l2norm_fwd", "l2norm_bwd") if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"{example_path} does not expose: {', '.join(missing)}")
    return module.l2norm_fwd, module.l2norm_bwd


def check_npu_runtime() -> None:
    torch = cmp.torch
    try:
        import torch_npu  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import
    except ImportError as exc:
        raise RuntimeError(f"torch_npu is required: {exc}") from exc
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU is unavailable.")


def python_l2norm(x, eps: float = 1e-6):
    torch = cmp.torch
    rstd = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
    return (x * rstd).to(x.dtype), rstd


def broadcast_rstd(rstd, target):
    while rstd.ndim < target.ndim:
        rstd = rstd.unsqueeze(-1)
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


def make_x(case: cmp.Case, device, dtype, seed: int, layout: str):
    q, _, _, _, _, _ = cmp.make_inputs(case, device, dtype, seed)
    if layout == "BHTD":
        return q.transpose(1, 2).contiguous()
    if layout == "BTHD":
        return q.contiguous()
    raise ValueError(f"unsupported layout: {layout}")


def token_dim(layout: str) -> int:
    return 2 if layout == "BHTD" else 1


def head_dim(layout: str) -> int:
    return 1 if layout == "BHTD" else 2


def segment_lengths(case: cmp.Case) -> list[int]:
    if case.cu_seqlens is None:
        return [case.seq_len]
    return [b - a for a, b in zip(case.cu_seqlens, case.cu_seqlens[1:])]


def segment_tail_mask(case: cmp.Case, layout: str, device):
    torch = cmp.torch
    tails = [length % case.chunk_size for length in segment_lengths(case)]
    if not any(tails):
        return None

    if layout == "BHTD":
        mask = torch.zeros(case.batch, case.heads, case.seq_len, device=device, dtype=torch.bool)
    else:
        mask = torch.zeros(case.batch, case.seq_len, case.heads, device=device, dtype=torch.bool)

    starts = [0]
    if case.cu_seqlens is not None:
        starts = list(case.cu_seqlens[:-1])

    for start, length, rem in zip(starts, segment_lengths(case), tails):
        if rem == 0:
            continue
        tail_start = start + length - rem
        tail_end = start + length
        if layout == "BHTD":
            mask[:, :, tail_start:tail_end] = True
        else:
            mask[:, tail_start:tail_end, :] = True
    return mask


def tensor_stats(actual, expected, args: argparse.Namespace, mask=None) -> dict[str, Any]:
    return cmp.tensor_stats(actual, expected, args.atol, args.rtol, mask)


def topk_errors(actual, expected, args: argparse.Namespace, mask=None) -> list[dict[str, Any]]:
    torch = cmp.torch
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    if actual_f.shape != expected_f.shape:
        return []
    diff = (actual_f - expected_f).abs()
    if mask is not None:
        mask = mask.to(device=diff.device, dtype=torch.bool)
        while mask.ndim < diff.ndim:
            mask = mask.unsqueeze(-1)
        diff = diff.masked_fill(~mask.expand_as(diff), -1)
    flat = diff.flatten()
    if flat.numel() == 0:
        return []
    k = min(args.tail_topk, flat.numel())
    vals, idxs = torch.topk(flat, k)
    out = []
    for val, idx in zip(vals.detach().cpu().tolist(), idxs.detach().cpu().tolist()):
        if val < 0:
            continue
        coord = list(torch.unravel_index(torch.tensor(idx, device=diff.device), diff.shape))
        out.append(
            {
                "abs_diff": float(val),
                "index": [int(c.detach().cpu().item()) for c in coord],
            }
        )
    return out


def run_case(case: cmp.Case, args: argparse.Namespace, l2norm_fwd, l2norm_bwd) -> dict[str, Any]:
    torch = cmp.torch
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(args.device)
    if hasattr(torch.npu, "set_compile_mode"):
        torch.npu.set_compile_mode(jit_compile=False)

    x = make_x(case, device, dtype, args.seed, args.layout)
    x_ref = cmp.clone_leaf(x)
    x_kernel = x.detach().clone()

    torch.manual_seed(args.seed + 17)
    dy = torch.randn_like(x_ref)

    y_ref, rstd_ref = python_l2norm(x_ref)
    y_ref.backward(dy)
    dx_ref = x_ref.grad

    y_kernel, rstd_kernel = l2norm_fwd(x_kernel)
    rstd_kernel_b = broadcast_rstd(rstd_kernel, x_kernel)
    dx_kernel_with_normalized = l2norm_bwd(y_kernel, rstd_kernel, dy)
    dx_kernel_with_original = l2norm_bwd(x_kernel, rstd_kernel, dy)
    torch.npu.synchronize()

    dx_py_normalized = py_l2norm_bwd_normalized(y_kernel, rstd_kernel, dy)
    dx_py_original = py_l2norm_bwd_original(x_kernel, rstd_kernel, dy)
    tail_mask = segment_tail_mask(case, args.layout, device)

    comparisons = {
        "fwd_output": tensor_stats(y_kernel, y_ref, args),
        "fwd_rstd": tensor_stats(rstd_kernel_b, rstd_ref, args),
        "bwd_kernel_normalized_vs_autograd": tensor_stats(dx_kernel_with_normalized, dx_ref, args),
        "bwd_kernel_original_vs_autograd": tensor_stats(dx_kernel_with_original, dx_ref, args),
        "bwd_kernel_normalized_vs_py_formula": tensor_stats(dx_kernel_with_normalized, dx_py_normalized, args),
        "bwd_kernel_original_vs_py_formula": tensor_stats(dx_kernel_with_original, dx_py_original, args),
        "bwd_py_normalized_vs_autograd": tensor_stats(dx_py_normalized, dx_ref, args),
        "bwd_py_original_vs_autograd": tensor_stats(dx_py_original, dx_ref, args),
    }
    tail_reports = {}
    if tail_mask is not None:
        tail_reports = {
            "bwd_kernel_normalized_vs_autograd": tensor_stats(dx_kernel_with_normalized, dx_ref, args, tail_mask),
            "bwd_kernel_original_vs_autograd": tensor_stats(dx_kernel_with_original, dx_ref, args, tail_mask),
            "bwd_py_normalized_vs_autograd": tensor_stats(dx_py_normalized, dx_ref, args, tail_mask),
            "bwd_py_original_vs_autograd": tensor_stats(dx_py_original, dx_ref, args, tail_mask),
        }

    return {
        "case": case.__dict__,
        "shape": list(x.shape),
        "layout": args.layout,
        "token_dim": token_dim(args.layout),
        "head_dim": head_dim(args.layout),
        "dtype": args.dtype,
        "device": str(device),
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "rstd_shape": list(rstd_kernel.shape),
        "segment_lengths": segment_lengths(case),
        "segment_tails": [length % case.chunk_size for length in segment_lengths(case)],
        "passed": (
            comparisons["fwd_output"]["allclose"]
            and comparisons["fwd_rstd"]["allclose"]
            and (
                comparisons["bwd_kernel_normalized_vs_autograd"]["allclose"]
                or comparisons["bwd_kernel_original_vs_autograd"]["allclose"]
            )
        ),
        "comparisons": comparisons,
        "tail_reports": tail_reports,
        "topk": {
            "normalized": topk_errors(dx_kernel_with_normalized, dx_ref, args),
            "original": topk_errors(dx_kernel_with_original, dx_ref, args),
            "tail_normalized": topk_errors(dx_kernel_with_normalized, dx_ref, args, tail_mask)
            if tail_mask is not None
            else [],
            "tail_original": topk_errors(dx_kernel_with_original, dx_ref, args, tail_mask)
            if tail_mask is not None
            else [],
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
        check_npu_runtime()
        fla_repo = find_fla_repo(args.fla_repo)
        l2norm_fwd, l2norm_bwd = load_l2norm_kernels(fla_repo)
        case = cmp.override_case(cmp.CASES[args.case], args)
        result = run_case(case, args, l2norm_fwd, l2norm_bwd)
    except Exception as exc:  # pylint: disable=broad-except
        payload = {
            "comparison": "real_l2norm_kernel_vs_pytorch_autograd",
            "error": str(exc),
            "passed": False,
        }
        write_payload(payload, args.output_json)
        return 2

    payload = {
        "comparison": "real_l2norm_kernel_vs_pytorch_autograd",
        "fla_repo": str(fla_repo),
        "torch": torch.__version__,
        **result,
    }
    write_payload(payload, args.output_json)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
