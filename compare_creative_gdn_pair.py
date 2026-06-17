#!/usr/bin/env python3
"""Compare creative pure Triton GDN against creative AscendC-mixed GDN.

This script intentionally compares the two implementations from the creative
repo itself:

  * pure Triton: mindspeed_mm/fsdp/models/qwen3_5/chunk_gated_delta_rule.py
  * AscendC mixed: mindspeed_mm/fsdp/models/qwen3_5/flash_gated_delta_rule.py

It does not use flash-linear-attention-npu/examples as a substitute wrapper.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import compare_gdn_precision as cmp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare creative pure Triton GDN with creative AscendC-mixed GDN."
    )
    parser.add_argument("--case", choices=cmp.CASES.keys(), default="varlen")
    parser.add_argument(
        "--creative-repo",
        default=os.environ.get("CREATIVE_REPO"),
        help="Optional path to qwen3.5_omni_creative checkout. Defaults to the vendored creative_snapshot.",
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
    parser.add_argument("--tail-topk", type=int, default=8)
    parser.add_argument(
        "--no-mindspeed-triton-shim",
        action="store_true",
        help=(
            "Do not shim mindspeed.lite.ops.triton imports to the selected "
            "snapshot/repo qwen3_5/triton modules when the external mindspeed package is missing."
        ),
    )
    return parser.parse_args()


def find_creative_repo(path_arg: str | None) -> Path:
    candidates = []
    if path_arg:
        candidates.append(Path(path_arg))
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            script_dir / "creative_snapshot",
            cwd / "creative_snapshot",
            cwd,
            cwd.parent / "qwen3.5_omni_creative",
            cwd.parent / "MindSpeed-MM",
        ]
    )
    for candidate in candidates:
        marker = candidate / "mindspeed_mm" / "fsdp" / "models" / "qwen3_5" / "flash_gated_delta_rule.py"
        if marker.is_file():
            return candidate.resolve()
    raise RuntimeError(
        "Cannot find creative snapshot or repo. Keep creative_snapshot in this test repo, "
        "or pass --creative-repo /path/to/qwen3.5_omni_creative."
    )


def ensure_runtime_for_mixed_ops() -> None:
    cmp.import_torch_runtime()
    cmp.check_npu_runtime()


def make_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_creative_file_package(creative_repo: Path) -> str:
    """Create a synthetic package for qwen3_5 files without importing mindspeed_mm."""
    qwen_dir = creative_repo / "mindspeed_mm" / "fsdp" / "models" / "qwen3_5"
    triton_dir = qwen_dir / "triton"
    package = "_creative_qwen3_5_under_test"

    pkg = types.ModuleType(package)
    pkg.__path__ = [str(qwen_dir)]  # type: ignore[attr-defined]
    sys.modules[package] = pkg

    triton_pkg = types.ModuleType(f"{package}.triton")
    triton_pkg.__path__ = [str(triton_dir)]  # type: ignore[attr-defined]
    sys.modules[f"{package}.triton"] = triton_pkg
    return package


def install_mindspeed_triton_shim(package: str) -> None:
    """Map mindspeed.lite.ops.triton.* to the synthetic creative Triton package."""

    torch = cmp.torch

    sys.modules.setdefault("mindspeed", make_module("mindspeed"))
    sys.modules.setdefault("mindspeed.lite", make_module("mindspeed.lite"))
    sys.modules.setdefault("mindspeed.lite.ops", make_module("mindspeed.lite.ops"))
    sys.modules.setdefault("mindspeed.lite.ops.triton", make_module("mindspeed.lite.ops.triton"))

    for short in ("chunk_scaled_dot_kkt", "wy_fast", "solve_tril", "cumsum", "utils"):
        sys.modules[f"mindspeed.lite.ops.triton.{short}"] = importlib.import_module(f"{package}.triton.{short}")

    l2norm_module = types.ModuleType("mindspeed.lite.ops.triton.l2norm")

    def l2norm_fwd(x, eps: float = 1e-6):
        original_dtype = x.dtype
        rstd = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
        return (x * rstd).to(original_dtype), rstd

    def l2norm_bwd(x, rstd, dy):
        y = (x * rstd).to(x.dtype)
        dot = (dy * y).sum(dim=-1, keepdim=True)
        return ((dy - y * dot) * rstd).to(x.dtype)

    l2norm_module.l2norm_fwd = l2norm_fwd
    l2norm_module.l2norm_bwd = l2norm_bwd
    sys.modules["mindspeed.lite.ops.triton.l2norm"] = l2norm_module


def import_creative_pair(creative_repo: Path, allow_shim: bool):
    package = install_creative_file_package(creative_repo)
    qwen_dir = creative_repo / "mindspeed_mm" / "fsdp" / "models" / "qwen3_5"

    shim_used = False
    pure_module = load_module_from_path(
        f"{package}.chunk_gated_delta_rule",
        qwen_dir / "chunk_gated_delta_rule.py",
    )
    try:
        mixed_module = load_module_from_path(
            f"{package}.flash_gated_delta_rule",
            qwen_dir / "flash_gated_delta_rule.py",
        )
    except ModuleNotFoundError as exc:
        if not allow_shim or not (exc.name or "").startswith("mindspeed"):
            raise
        install_mindspeed_triton_shim(package)
        shim_used = True
        mixed_module = load_module_from_path(
            f"{package}.flash_gated_delta_rule",
            qwen_dir / "flash_gated_delta_rule.py",
        )

    return pure_module.chunk_gated_delta_rule, mixed_module.flash_gated_delta_rule, shim_used


def clone_inputs(inputs: tuple[Any, ...]):
    q, k, v, beta, g, cu = inputs
    return [cmp.clone_leaf(x) for x in (q, k, v, beta, g)], cu


def maybe_unflatten(cu, x):
    return cmp.varlen_to_nonvarlen(cu, x) if cu is not None else x


def packed_tail_mask(case: cmp.Case, cu, device):
    torch = cmp.torch
    tail = case.seq_len % case.chunk_size
    if tail == 0:
        return None
    start = case.seq_len - tail
    end = case.seq_len
    if cu is None:
        mask = torch.zeros(case.batch, case.seq_len, device=device, dtype=torch.bool)
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
    return mask if bool(mask.any().item()) else None


def valid_mask(case: cmp.Case, cu, device):
    torch = cmp.torch
    if cu is None:
        return torch.ones(case.batch, case.seq_len, device=device, dtype=torch.bool)
    return cmp.varlen_valid_mask(cu, device)


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
    expanded = mask.to(device=actual_f.device, dtype=torch.bool)
    while expanded.ndim < actual_f.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(actual_f)
    diff = (actual_f - expected_f).abs()
    selected_count = int(expanded.sum().item())
    if selected_count == 0:
        return []
    values, flat_indices = torch.topk(diff.masked_fill(~expanded, -1).flatten(), k=min(limit, selected_count))
    rows = []
    shape = actual_f.shape
    for value, flat_index in zip(values.detach().cpu().tolist(), flat_indices.detach().cpu().tolist()):
        if value < 0:
            continue
        coords = []
        rem = int(flat_index)
        for dim in reversed(shape):
            coords.append(rem % dim)
            rem //= dim
        coords = list(reversed(coords))
        seq_idx, tok_idx = coords[0], coords[1]
        h_idx = coords[2] if len(coords) > 2 else None
        d_idx = coords[3] if len(coords) > 3 else None
        actual_value = actual_f[tuple(coords)].item()
        expected_value = expected_f[tuple(coords)].item()
        rows.append(
            {
                "seq": int(seq_idx),
                "token": int(tok_idx),
                "packed_token": int(packed_index_for_token(cu, seq_idx, tok_idx, case.seq_len)),
                "head": None if h_idx is None else int(h_idx),
                "dim": None if d_idx is None else int(d_idx),
                "actual": float(actual_value),
                "expected": float(expected_value),
                "abs_diff": float(value),
                "rel_diff": float(value / max(abs(expected_value), 1e-12)),
            }
        )
    return rows


def compare_tensors(mixed: dict[str, Any], pure: dict[str, Any], case: cmp.Case, cu, args: argparse.Namespace):
    device = mixed["grad_k"].device
    mask = cmp.varlen_valid_mask(cu, device) if cu is not None else None
    comparisons = {}
    for name in ("output", "grad_q", "grad_k", "grad_v", "grad_beta", "grad_g"):
        comparisons[name] = cmp.tensor_stats(
            maybe_unflatten(cu, mixed[name]),
            maybe_unflatten(cu, pure[name]),
            args.atol,
            args.rtol,
            mask,
        )

    tail = packed_tail_mask(case, cu, device)
    tail_report = None
    if tail is not None:
        mixed_k = maybe_unflatten(cu, mixed["grad_k"])
        pure_k = maybe_unflatten(cu, pure["grad_k"])
        tail_report = {
            "token_count": int(tail.sum().item()),
            "packed_range": [case.seq_len - case.seq_len % case.chunk_size, case.seq_len],
            "stats": cmp.tensor_stats(mixed_k, pure_k, args.atol, args.rtol, tail),
            "top_errors": topk_errors(mixed_k, pure_k, tail, cu, case, args.tail_topk),
        }
        non_tail = valid_mask(case, cu, device) & ~tail
        if bool(non_tail.any().item()):
            tail_report["non_tail_stats"] = cmp.tensor_stats(mixed_k, pure_k, args.atol, args.rtol, non_tail)
    return comparisons, tail_report


def run_impl(fn, inputs: tuple[Any, ...], do, case: cmp.Case, use_qk_l2norm: bool):
    torch = cmp.torch
    leaves, cu = clone_inputs(inputs)
    q, k, v, beta, g = leaves
    out, _ = fn(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu,
        chunk_size=case.chunk_size,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )
    out.backward(do)
    torch.npu.synchronize()
    return {
        "output": out,
        "grad_q": q.grad,
        "grad_k": k.grad,
        "grad_v": v.grad,
        "grad_beta": beta.grad,
        "grad_g": g.grad,
    }


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


def run_case(case: cmp.Case, args: argparse.Namespace, pure_fn, mixed_fn, shim_used: bool, creative_repo: Path):
    torch = cmp.torch
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(args.device)
    if hasattr(torch.npu, "set_compile_mode"):
        torch.npu.set_compile_mode(jit_compile=False)

    inputs = cmp.make_inputs(case, device, dtype, args.seed)
    use_qk_l2norm = not args.no_qk_l2norm

    torch.manual_seed(args.seed + 1)
    do = torch.randn(
        case.batch,
        case.seq_len,
        case.heads,
        case.value_dim,
        dtype=dtype,
        device=device,
    )

    print(
        "running creative pair",
        f"case={case.name}",
        f"B={case.batch}",
        f"T={case.seq_len}",
        f"H={case.heads}",
        f"K={case.key_dim}",
        f"V={case.value_dim}",
        f"chunk_size={case.chunk_size}",
        f"varlen={case.cu_seqlens is not None}",
        flush=True,
    )

    pure = run_impl(pure_fn, inputs, do, case, use_qk_l2norm)
    mixed = run_impl(mixed_fn, inputs, do, case, use_qk_l2norm)
    comparisons, tail_report = compare_tensors(mixed, pure, case, inputs[-1], args)
    failed = [name for name, stats in comparisons.items() if not stats["allclose"]]
    return {
        "case": case.__dict__,
        "shape_summary": segment_summary(case),
        "dtype": args.dtype,
        "device": str(device),
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "use_qk_l2norm_in_kernel": use_qk_l2norm,
        "creative_repo": str(creative_repo),
        "mindspeed_triton_shim_used": shim_used,
        "baseline": "creative_pure_triton_file: mindspeed_mm/fsdp/models/qwen3_5/chunk_gated_delta_rule.py",
        "actual": "creative_ascendc_mixed_file: mindspeed_mm/fsdp/models/qwen3_5/flash_gated_delta_rule.py",
        "passed": not failed,
        "failed_tensors": failed,
        "comparisons": comparisons,
        "tail_reports": {"packed_final_tail_grad_k": tail_report} if tail_report else {},
    }


def main() -> int:
    args = parse_args()
    try:
        torch, _ = cmp.import_torch_runtime()
        creative_repo = find_creative_repo(args.creative_repo)
        ensure_runtime_for_mixed_ops()
        pure_fn, mixed_fn, shim_used = import_creative_pair(
            creative_repo, allow_shim=not args.no_mindspeed_triton_shim
        )
        case = cmp.override_case(cmp.CASES[args.case], args)
        result = run_case(case, args, pure_fn, mixed_fn, shim_used, creative_repo)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    payload = {
        "comparison": "creative_ascendc_mixed_vs_creative_pure_triton",
        "torch": torch.__version__,
        "passed": result["passed"],
        "cases": [result],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
