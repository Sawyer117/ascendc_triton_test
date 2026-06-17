#!/usr/bin/env python3
"""Compare FLA-npu AscendC GDN against a pure PyTorch reference.

This script intentionally does not import MindSpeed-MM. It loads the AscendC
GDN wrapper from a local flash-linear-attention-npu checkout and compares it
with an in-file torch reference on identical inputs.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
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
    # FLA-npu GDN AscendC kernels are validated around K=128 and V=128/256.
    # Smaller head dims such as K=64/V=64 can fail host-side tiling before any precision comparison runs.
    "small": Case("small", 1, 256, 4, 128, 128, 64),
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


REQUIRED_NPU_OPS = (
    "npu_recompute_w_u_fwd",
    "npu_chunk_gated_delta_rule_fwd_h",
    "npu_chunk_fwd_o",
    "npu_chunk_bwd_dv_local",
    "npu_chunk_gated_delta_rule_bwd_dhu",
    "npu_chunk_bwd_dqkwg",
    "npu_prepare_wy_repr_bwd_da",
    "npu_prepare_wy_repr_bwd_full",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FLA-npu AscendC GDN with a pure PyTorch reference."
    )
    parser.add_argument("--case", choices=[*CASES.keys(), "all"], default="small")
    parser.add_argument("--impl", choices=["fla", "creative"], default=os.environ.get("GDN_IMPL", "fla"))
    parser.add_argument(
        "--fla-repo",
        default=os.environ.get("FLA_NPU_REPO"),
        help="Path to flash-linear-attention-npu. Defaults to ./flash-linear-attention-npu if present.",
    )
    parser.add_argument(
        "--creative-repo",
        default=os.environ.get("CREATIVE_REPO"),
        help="Optional creative checkout for --impl creative. Defaults to the vendored creative_snapshot.",
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
    return parser.parse_args()


def import_torch_runtime():
    global torch, F  # pylint: disable=global-statement
    if torch is None:
        import torch as torch_module  # pylint: disable=import-outside-toplevel
        import torch.nn.functional as functional  # pylint: disable=import-outside-toplevel

        torch = torch_module
        F = functional
    return torch, F


def find_fla_repo(path_arg: str | None) -> Path:
    candidates = []
    if path_arg:
        candidates.append(Path(path_arg))
    cwd = Path.cwd()
    candidates.extend(
        [
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
        "Cannot find flash-linear-attention-npu. Pass --fla-repo /path/to/flash-linear-attention-npu "
        "or run from a directory containing ./flash-linear-attention-npu."
    )


def check_npu_runtime():
    try:
        import torch_npu  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import
    except ImportError as exc:
        raise RuntimeError("torch_npu is required.") from exc

    try:
        import fla_npu  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            "fla_npu failed to import. Build/install flash-linear-attention-npu/torch_custom/fla_npu, "
            "run this script outside torch_custom/fla_npu so the installed wheel is not shadowed, "
            "and export the custom op_api lib path in LD_LIBRARY_PATH."
        ) from exc

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU is not available. Source CANN env and run on an Ascend host.")

    missing_ops = [op for op in REQUIRED_NPU_OPS if not hasattr(torch.ops.npu, op)]
    if missing_ops:
        raise RuntimeError(
            "Missing FLA-npu custom ops: "
            + ", ".join(missing_ops)
            + ". Reinstall the FLA-npu .run package and torch_custom/fla_npu wheel in the same environment."
        )


def load_flash_gdn(fla_repo: Path):
    import_torch_runtime()
    check_npu_runtime()

    if str(fla_repo) not in sys.path:
        sys.path.insert(0, str(fla_repo))

    example_path = fla_repo / "examples" / "flash_gated_delta_rule.py"
    spec = importlib.util.spec_from_file_location("fla_npu_flash_gated_delta_rule_example", example_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {example_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.flash_gated_delta_rule


def find_creative_repo(path_arg: str | None) -> Path | None:
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
        if (candidate / "mindspeed_mm" / "fsdp" / "models" / "qwen3_5" / "flash_gated_delta_rule.py").is_file():
            return candidate.resolve()
    return None


def _make_module(name: str):
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _install_creative_package(creative_repo: Path) -> str:
    qwen_dir = creative_repo / "mindspeed_mm" / "fsdp" / "models" / "qwen3_5"
    triton_dir = qwen_dir / "triton"
    package = "_creative_qwen3_5_cmp"
    pkg = types.ModuleType(package)
    pkg.__path__ = [str(qwen_dir)]  # type: ignore[attr-defined]
    sys.modules[package] = pkg
    triton_pkg = types.ModuleType(f"{package}.triton")
    triton_pkg.__path__ = [str(triton_dir)]  # type: ignore[attr-defined]
    sys.modules[f"{package}.triton"] = triton_pkg
    return package


def _install_mindspeed_triton_shim(package: str) -> None:
    sys.modules.setdefault("mindspeed", _make_module("mindspeed"))
    sys.modules.setdefault("mindspeed.lite", _make_module("mindspeed.lite"))
    sys.modules.setdefault("mindspeed.lite.ops", _make_module("mindspeed.lite.ops"))
    sys.modules.setdefault("mindspeed.lite.ops.triton", _make_module("mindspeed.lite.ops.triton"))
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


def load_creative_gdn(creative_repo_arg: str | None):
    import_torch_runtime()
    check_npu_runtime()

    creative_repo = find_creative_repo(creative_repo_arg)
    if creative_repo is None:
        raise RuntimeError(
            "Cannot find creative implementation. Keep creative_snapshot in this test repo, "
            "or pass --creative-repo /path/to/qwen3.5_omni_creative."
        )
    package = _install_creative_package(creative_repo)
    module_name = f"{package}.flash_gated_delta_rule"
    module_path = creative_repo / "mindspeed_mm" / "fsdp" / "models" / "qwen3_5" / "flash_gated_delta_rule.py"
    try:
        module = _load_module_from_path(module_name, module_path)
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("mindspeed"):
            raise
        _install_mindspeed_triton_shim(package)
        module = _load_module_from_path(module_name, module_path)
    return module.flash_gated_delta_rule, creative_repo


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
    cases = list(CASES.values()) if args.case == "all" else [CASES[args.case]]
    for case in cases:
        yield override_case(case, args)


def l2norm(x, dim: int = -1, eps: float = 1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def varlen_to_nonvarlen(cu_seqlens, *inputs):
    cu = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    batch = len(cu) - 1
    max_len = max(cu[i + 1] - cu[i] for i in range(batch))
    outputs = [torch.zeros(batch, max_len, *x.shape[2:], device=x.device, dtype=x.dtype) for x in inputs]
    for i in range(batch):
        start, end = cu[i], cu[i + 1]
        if end > start:
            for out, x in zip(outputs, inputs):
                out[i, : end - start] = x[0, start:end]
    return outputs[0] if len(outputs) == 1 else outputs


def varlen_valid_mask(cu_seqlens, device):
    cu = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    batch = len(cu) - 1
    max_len = max(cu[i + 1] - cu[i] for i in range(batch))
    mask = torch.zeros(batch, max_len, device=device, dtype=torch.bool)
    for i in range(batch):
        mask[i, : cu[i + 1] - cu[i]] = True
    return mask


def ref_torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size: int = 64,
    initial_state=None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = query.shape[-1] ** -0.5
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def make_inputs(case: Case, device, dtype, seed: int):
    if case.cu_seqlens is not None:
        if case.batch != 1:
            raise ValueError("Packed varlen inputs require batch=1.")
        if case.cu_seqlens[-1] != case.seq_len:
            raise ValueError("Last cu_seqlens value must equal seq_len.")
    torch.manual_seed(seed)
    q = torch.randn(case.batch, case.seq_len, case.heads, case.key_dim, dtype=dtype, device=device)
    k = torch.randn(case.batch, case.seq_len, case.heads, case.key_dim, dtype=dtype, device=device)
    v = torch.randn(case.batch, case.seq_len, case.heads, case.value_dim, dtype=dtype, device=device)
    beta = torch.rand(case.batch, case.seq_len, case.heads, dtype=dtype, device=device).sigmoid()
    g = F.logsigmoid(torch.rand(case.batch, case.seq_len, case.heads, dtype=dtype, device=device))
    cu = None
    if case.cu_seqlens is not None:
        cu = torch.tensor(case.cu_seqlens, dtype=torch.long, device=device)
    return q, k, v, beta, g, cu


def clone_leaf(x):
    return x.detach().clone().requires_grad_(True)


def tensor_stats(actual, expected, atol: float, rtol: float, mask=None) -> dict[str, object]:
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    if actual_f.shape != expected_f.shape:
        raise ValueError(f"shape mismatch: actual={tuple(actual_f.shape)}, expected={tuple(expected_f.shape)}")

    if mask is not None:
        mask = mask.to(device=actual_f.device, dtype=torch.bool)
        while mask.ndim < actual_f.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(actual_f)
        actual_f = actual_f[mask]
        expected_f = expected_f[mask]
        if actual_f.numel() == 0:
            raise ValueError("empty comparison mask")

    diff = actual_f - expected_f
    abs_diff = diff.abs()
    denom = expected_f.abs().clamp_min(1e-12)
    rel_diff = abs_diff / denom
    tolerance = atol + rtol * expected_f.abs()
    mismatches = abs_diff > tolerance
    rms = torch.sqrt(torch.mean(diff * diff))
    return {
        "allclose": bool(torch.allclose(actual_f, expected_f, atol=atol, rtol=rtol)),
        "numel": int(actual_f.numel()),
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "rms": float(rms.item()),
        "max_rel": float(rel_diff.max().item()),
        "mismatch_ratio": float(mismatches.float().mean().item()),
    }


def run_one_case(case: Case, args: argparse.Namespace, flash_gdn) -> dict[str, object]:
    print(
        "running",
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
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(args.device)
    if hasattr(torch.npu, "set_compile_mode"):
        torch.npu.set_compile_mode(jit_compile=False)

    q, k, v, beta, g, cu = make_inputs(case, device, dtype, args.seed)
    q_ref, k_ref, v_ref, beta_ref, g_ref = [clone_leaf(x) for x in (q, k, v, beta, g)]
    if args.impl == "fla":
        q_ac, k_ac, v_ac = [clone_leaf(x.transpose(1, 2).contiguous()) for x in (q, k, v)]
    else:
        q_ac, k_ac, v_ac = [clone_leaf(x) for x in (q, k, v)]
    beta_ac, g_ac = [clone_leaf(x) for x in (beta, g)]
    use_qk_l2norm = not args.no_qk_l2norm

    o_ac, _ = flash_gdn(
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

    if cu is not None:
        q_ref_in, k_ref_in, v_ref_in, beta_ref_in, g_ref_in = varlen_to_nonvarlen(
            cu, q_ref, k_ref, v_ref, beta_ref, g_ref
        )
        q_ref_in.retain_grad()
        k_ref_in.retain_grad()
        v_ref_in.retain_grad()
        beta_ref_in.retain_grad()
        g_ref_in.retain_grad()
    else:
        q_ref_in, k_ref_in, v_ref_in, beta_ref_in, g_ref_in = q_ref, k_ref, v_ref, beta_ref, g_ref

    o_ref, _ = ref_torch_chunk_gated_delta_rule(
        q_ref_in,
        k_ref_in,
        v_ref_in,
        g_ref_in,
        beta_ref_in,
        chunk_size=case.chunk_size,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )

    torch.manual_seed(args.seed + 1)
    do_ac = torch.randn(o_ac.shape, dtype=o_ac.dtype, device=device)
    do_ref = varlen_to_nonvarlen(cu, do_ac) if cu is not None else do_ac
    o_ac.backward(do_ac)
    o_ref.backward(do_ref)
    torch.npu.synchronize()

    if args.impl == "fla":
        q_ac_grad, k_ac_grad, v_ac_grad = [x.grad.transpose(1, 2).contiguous() for x in (q_ac, k_ac, v_ac)]
    else:
        q_ac_grad, k_ac_grad, v_ac_grad = [x.grad for x in (q_ac, k_ac, v_ac)]

    if cu is not None:
        valid_mask = varlen_valid_mask(cu, device)
        o_ac_cmp = varlen_to_nonvarlen(cu, o_ac)
        q_ac_g, k_ac_g, v_ac_g = [varlen_to_nonvarlen(cu, x) for x in (q_ac_grad, k_ac_grad, v_ac_grad)]
        beta_ac_g, g_ac_g = varlen_to_nonvarlen(cu, beta_ac.grad, g_ac.grad)
        q_ref_g, k_ref_g, v_ref_g, beta_ref_g, g_ref_g = (
            q_ref_in.grad,
            k_ref_in.grad,
            v_ref_in.grad,
            beta_ref_in.grad,
            g_ref_in.grad,
        )
    else:
        valid_mask = None
        o_ac_cmp = o_ac
        q_ac_g, k_ac_g, v_ac_g = q_ac_grad, k_ac_grad, v_ac_grad
        beta_ac_g, g_ac_g = beta_ac.grad, g_ac.grad
        q_ref_g, k_ref_g, v_ref_g, beta_ref_g, g_ref_g = q_ref.grad, k_ref.grad, v_ref.grad, beta_ref.grad, g_ref.grad

    comparisons = {
        "output": tensor_stats(o_ac_cmp, o_ref, args.atol, args.rtol, valid_mask),
        "grad_q": tensor_stats(q_ac_g, q_ref_g, args.atol, args.rtol, valid_mask),
        "grad_k": tensor_stats(k_ac_g, k_ref_g, args.atol, args.rtol, valid_mask),
        "grad_v": tensor_stats(v_ac_g, v_ref_g, args.atol, args.rtol, valid_mask),
        "grad_beta": tensor_stats(beta_ac_g, beta_ref_g, args.atol, args.rtol, valid_mask),
        "grad_g": tensor_stats(g_ac_g, g_ref_g, args.atol, args.rtol, valid_mask),
    }
    return {
        "case": case.__dict__,
        "impl": args.impl,
        "dtype": args.dtype,
        "device": str(device),
        "seed": args.seed,
        "atol": args.atol,
        "rtol": args.rtol,
        "use_qk_l2norm_in_kernel": use_qk_l2norm,
        "passed": all(item["allclose"] for item in comparisons.values()),
        "comparisons": comparisons,
    }


def main() -> int:
    args = parse_args()
    try:
        import_torch_runtime()
        creative_repo = None
        fla_repo = None
        if args.impl == "fla":
            fla_repo = find_fla_repo(args.fla_repo)
            flash_gdn = load_flash_gdn(fla_repo)
        else:
            flash_gdn, creative_repo = load_creative_gdn(args.creative_repo)
        results = [run_one_case(case, args, flash_gdn) for case in selected_cases(args)]
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    payload = {
        "impl": args.impl,
        "fla_repo": str(fla_repo) if fla_repo is not None else None,
        "creative_repo": str(creative_repo) if creative_repo is not None else None,
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
