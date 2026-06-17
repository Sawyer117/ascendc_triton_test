#!/usr/bin/env python3
"""Compare Triton and AscendC GDN kernels on identical inputs.

This script is intended to run on an Ascend NPU environment with both the
Triton-on-NPU path and the FLA-npu AscendC custom ops installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Iterable

torch = None
F = None


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    seq_len: int
    heads: int
    key_dim: int
    value_dim: int
    chunk_size: int
    cu_seqlens: tuple[int, ...] | None = None


CASES = {
    "small": Case("small", 1, 128, 4, 64, 64, 64),
    "medium": Case("medium", 1, 1024, 32, 128, 128, 64),
    "varlen": Case(
        "varlen",
        1,
        1121,
        8,
        128,
        128,
        64,
        (0, 112, 209, 240, 281, 489, 523, 566, 689, 721, 785, 837, 985, 1071, 1121),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare qwen3.5 GDN Triton and AscendC implementations."
    )
    parser.add_argument(
        "--case",
        choices=[*CASES.keys(), "all"],
        default="small",
        help="Predefined input shape to run.",
    )
    parser.add_argument("--batch", type=int, help="Override batch size.")
    parser.add_argument("--seq-len", type=int, help="Override sequence length.")
    parser.add_argument("--heads", type=int, help="Override number of heads.")
    parser.add_argument("--key-dim", type=int, help="Override key/query head dim.")
    parser.add_argument("--value-dim", type=int, help="Override value head dim.")
    parser.add_argument("--chunk-size", type=int, help="Override chunk size.")
    parser.add_argument(
        "--cu-seqlens",
        type=str,
        help="Comma-separated cumulative sequence lengths for varlen mode. Batch must be 1.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument(
        "--no-qk-l2norm",
        action="store_true",
        help="Disable in-kernel q/k L2 normalization.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Optional path to write the full comparison result as JSON.",
    )
    return parser.parse_args()


def import_torch_runtime():
    global torch, F  # pylint: disable=global-statement
    if torch is None:
        import torch as torch_module  # pylint: disable=import-outside-toplevel
        import torch.nn.functional as functional  # pylint: disable=import-outside-toplevel

        torch = torch_module
        F = functional
    return torch, F


def require_runtime():
    import_torch_runtime()

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("torch_npu is required to run Ascend NPU kernels.") from exc

    try:
        import fla_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "fla_npu is required for AscendC GDN kernels. Build and install the "
            "FLA-npu custom op package first."
        ) from exc

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU is not available. Run this script on an Ascend NPU host.")

    from mindspeed_mm.fsdp.models.qwen3_5.chunk_gated_delta_rule import (  # pylint: disable=import-outside-toplevel
        chunk_gated_delta_rule as triton_gdn,
    )
    from mindspeed_mm.fsdp.models.qwen3_5.flash_gated_delta_rule import (  # pylint: disable=import-outside-toplevel
        flash_gated_delta_rule as ascendc_gdn,
    )

    return triton_gdn, ascendc_gdn


def parse_cu_seqlens(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    cu = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(cu) < 2 or cu[0] != 0:
        raise ValueError("--cu-seqlens must start with 0 and contain at least two values.")
    if any(b <= a for a, b in zip(cu, cu[1:])):
        raise ValueError("--cu-seqlens must be strictly increasing.")
    return cu


def override_case(base: Case, args: argparse.Namespace) -> Case:
    cu_seqlens = parse_cu_seqlens(args.cu_seqlens)
    return Case(
        name=base.name,
        batch=args.batch or base.batch,
        seq_len=args.seq_len or (cu_seqlens[-1] if cu_seqlens else base.seq_len),
        heads=args.heads or base.heads,
        key_dim=args.key_dim or base.key_dim,
        value_dim=args.value_dim or base.value_dim,
        chunk_size=args.chunk_size or base.chunk_size,
        cu_seqlens=cu_seqlens if cu_seqlens is not None else base.cu_seqlens,
    )


def selected_cases(args: argparse.Namespace) -> Iterable[Case]:
    if args.case == "all":
        cases = list(CASES.values())
    else:
        cases = [CASES[args.case]]
    for case in cases:
        yield override_case(case, args)


def clone_leaf(x: torch.Tensor) -> torch.Tensor:
    return x.detach().clone().requires_grad_(True)


def make_inputs(case: Case, device: torch.device, seed: int):
    if case.cu_seqlens is not None:
        if case.batch != 1:
            raise ValueError("Packed varlen inputs require batch=1.")
        if case.cu_seqlens[-1] != case.seq_len:
            raise ValueError("Last cu_seqlens value must equal seq_len.")

    torch.manual_seed(seed)
    dtype = torch.bfloat16
    shape_qk = (case.batch, case.seq_len, case.heads, case.key_dim)
    shape_v = (case.batch, case.seq_len, case.heads, case.value_dim)
    shape_g = (case.batch, case.seq_len, case.heads)

    q = torch.randn(shape_qk, dtype=dtype, device=device)
    k = torch.randn(shape_qk, dtype=dtype, device=device)
    v = torch.randn(shape_v, dtype=dtype, device=device)
    beta = torch.rand(shape_g, dtype=dtype, device=device).sigmoid()
    g = F.logsigmoid(torch.rand(shape_g, dtype=dtype, device=device))
    cu = None
    if case.cu_seqlens is not None:
        cu = torch.tensor(case.cu_seqlens, dtype=torch.long, device=device)
    return q, k, v, beta, g, cu


def tensor_stats(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> dict[str, object]:
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    diff = actual_f - expected_f
    abs_diff = diff.abs()
    denom = expected_f.abs().clamp_min(1e-12)
    rel_diff = abs_diff / denom
    tolerance = atol + rtol * expected_f.abs()
    mismatches = abs_diff > tolerance
    rms = torch.sqrt(torch.mean(diff * diff))
    return {
        "allclose": bool(torch.allclose(actual_f, expected_f, atol=atol, rtol=rtol)),
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "rms": float(rms.item()),
        "max_rel": float(rel_diff.max().item()),
        "mismatch_ratio": float(mismatches.float().mean().item()),
    }


def run_one_case(case: Case, args: argparse.Namespace, triton_gdn, ascendc_gdn) -> dict[str, object]:
    device = torch.device("npu")
    q, k, v, beta, g, cu = make_inputs(case, device, args.seed)

    q_tt, k_tt, v_tt, beta_tt, g_tt = [clone_leaf(x) for x in (q, k, v, beta, g)]
    q_ac, k_ac, v_ac, beta_ac, g_ac = [clone_leaf(x) for x in (q, k, v, beta, g)]
    use_qk_l2norm = not args.no_qk_l2norm

    o_tt, _ = triton_gdn(
        q=q_tt,
        k=k_tt,
        v=v_tt,
        g=g_tt,
        beta=beta_tt,
        cu_seqlens=cu,
        chunk_size=case.chunk_size,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )
    o_ac, _ = ascendc_gdn(
        q=q_ac,
        k=k_ac,
        v=v_ac,
        g=g_ac,
        beta=beta_ac,
        cu_seqlens=cu,
        chunk_size=case.chunk_size,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )

    torch.npu.synchronize()

    torch.manual_seed(args.seed + 1)
    do = torch.randn(o_tt.shape, dtype=o_tt.dtype, device=device)
    o_tt.backward(do)
    o_ac.backward(do.detach().clone())
    torch.npu.synchronize()

    comparisons = {
        "output": tensor_stats(o_ac, o_tt, args.atol, args.rtol),
        "grad_q": tensor_stats(q_ac.grad, q_tt.grad, args.atol, args.rtol),
        "grad_k": tensor_stats(k_ac.grad, k_tt.grad, args.atol, args.rtol),
        "grad_v": tensor_stats(v_ac.grad, v_tt.grad, args.atol, args.rtol),
        "grad_beta": tensor_stats(beta_ac.grad, beta_tt.grad, args.atol, args.rtol),
        "grad_g": tensor_stats(g_ac.grad, g_tt.grad, args.atol, args.rtol),
    }
    passed = all(item["allclose"] for item in comparisons.values())
    return {
        "case": case.__dict__,
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "use_qk_l2norm_in_kernel": use_qk_l2norm,
        "passed": passed,
        "comparisons": comparisons,
    }


def main() -> int:
    args = parse_args()
    try:
        triton_gdn, ascendc_gdn = require_runtime()
        results = [run_one_case(case, args, triton_gdn, ascendc_gdn) for case in selected_cases(args)]
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    payload = {
        "torch": torch.__version__,
        "cases": results,
        "passed": all(case["passed"] for case in results),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
