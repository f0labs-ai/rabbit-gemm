# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 F0 Labs

"""GPU checks for `sr_w8a16`. Run them after installing the package, on a GPU it was built for: `pytest -s`.

Every kernel family is compared with an exact reference on Qwen3-8B's projection shapes: an fp32 matmul of
the bf16 activations with the dequantized per-channel FP8 weights. `-s` prints the error table.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

# (N, K) of Qwen3-8B's fused projections.
QWEN3_8B = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (24576, 4096), "down": (4096, 12288)}
# Output rounding to bf16 alone is up to 2^-9 per element; 2^-8 relative L2 leaves room for summation order.
GEMV_REL_TOL = 2.0**-8


def _quantized(n: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-channel FP8 weights exactly as the plugin holds them: ([K, N] column-major view, [N, 1] scale,
    fp32 dequantized [N, K] reference)."""
    from vllm.model_executor.layers.quantization.online.fp8 import (  # noqa: PLC0415
        _fp8_channel_scale,
        _fp8_quant_per_channel,
    )

    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.02
    scale = _fp8_channel_scale(w.float().abs().amax(dim=-1, keepdim=True))
    q = _fp8_quant_per_channel(w, scale)
    return q.t(), scale, q.float() * scale


def _rel_err(y: torch.Tensor, ref: torch.Tensor) -> float:
    return float((y.float() - ref).norm() / ref.norm())


@pytest.mark.parametrize("proj", list(QWEN3_8B))
def test_every_kernel_matches_reference(proj: str) -> None:
    from sr_vllm.w8a16 import CUTLASS, GemvPolicy, _candidates, ensure_workspace, run_choice  # noqa: PLC0415

    n, k = QWEN3_8B[proj]
    weight, scale, dequant = _quantized(n, k, seed=1)
    ensure_workspace(n, k, 8, torch.device("cuda"))
    for m in (1, 8, 24, 32, 64):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        ref = x.float() @ dequant.t()
        cutlass_err = _rel_err(run_choice(CUTLASS, x, weight, scale, None), ref)
        for choice in _candidates(m, GemvPolicy.SR):
            err = _rel_err(run_choice(choice, x, weight, scale, None), ref)
            print(f"| {proj} | {m} | {choice} | {err:.2e} | {cutlass_err:.2e} |")
            assert err < GEMV_REL_TOL, f"{proj} M={m} {choice}: relative error {err:.2e}"
        # W8A8 rounds activations to e4m3 (3 mantissa bits); W8A16 must be the more accurate path.
        assert cutlass_err > GEMV_REL_TOL


def test_plan_follows_policy_and_buckets() -> None:
    from sr_vllm.w8a16 import CUTLASS, GemvPolicy, KernelFamily, build_plan  # noqa: PLC0415

    n, k = QWEN3_8B["o"]
    buckets = (1, 8, 32, 64)
    sr = build_plan(n, k, GemvPolicy.SR, buckets, torch.device("cuda"))
    assert all(c.family is not KernelFamily.CUTLASS_W8A8 for c in sr.choices)
    assert sr.choose(3) == sr.choices[1]  # rounds up to bucket 8
    assert sr.choose(65) == CUTLASS  # prefill-sized inputs never take a GEMV
    stock = build_plan(n, k, GemvPolicy.CUTLASS, buckets, torch.device("cuda"))
    assert set(stock.choices) == {CUTLASS}


def test_custom_op_compiles_without_guards_on_batch() -> None:
    """One dynamic-shape graph must still dispatch on the real M at run time, as in vLLM's compile."""
    from sr_vllm import w8a16  # noqa: PLC0415

    n, k = QWEN3_8B["qkv"]
    weight, scale, dequant = _quantized(n, k, seed=2)
    plan = w8a16.build_plan(n, k, w8a16.GemvPolicy.AUTO, (1, 8, 32), torch.device("cuda"))
    w8a16._PLANS[(n, k)] = plan

    def linear(x: torch.Tensor) -> torch.Tensor:
        return torch.ops.sr_vllm.w8a16_linear(x * 1.0, weight, scale, None)  # type: ignore[no-any-return]

    compiled = torch.compile(linear, fullgraph=True, dynamic=True)
    for m in (1, 8, 32, 100):  # 100 is past the last bucket: always CUTLASS W8A8
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        choice = plan.choose(m)
        y = compiled(x)
        # Bit-identical to the eager kernel the plan picks for this M: the compiled graph dispatched on it.
        assert torch.equal(y, w8a16.run_choice(choice, x, weight, scale, None)), f"M={m}: not {choice}"
        tol = 0.1 if choice == w8a16.CUTLASS else GEMV_REL_TOL
        err = _rel_err(y, x.float() @ dequant.t())
        assert math.isfinite(err) and err < tol, f"M={m} {choice}: relative error {err:.2e}"


def test_split_k_is_deterministic_and_matches_single_pass() -> None:
    """Split-K partials are reduced in split order by the last CTA: identical bits on every run."""
    from sr_vllm.w8a16 import KernelChoice, KernelFamily, ensure_workspace, run_choice  # noqa: PLC0415

    n, k = QWEN3_8B["down"]
    weight, scale, dequant = _quantized(n, k, seed=3)
    ensure_workspace(n, k, 8, torch.device("cuda"))
    x = torch.randn(32, k, device="cuda", dtype=torch.bfloat16)
    ref = x.float() @ dequant.t()
    whole = run_choice(KernelChoice(KernelFamily.SR_TENSOR_CORE_WIDE, 1, 1), x, weight, scale, None)
    for split in (2, 4, 8):
        choice = KernelChoice(KernelFamily.SR_TENSOR_CORE_WIDE, 1, split)
        first = run_choice(choice, x, weight, scale, None)
        second = run_choice(choice, x, weight, scale, None)
        assert torch.equal(first, second), f"split {split}: not bit-identical across runs"
        assert _rel_err(first, ref) < GEMV_REL_TOL
        assert _rel_err(first, whole.float()) < GEMV_REL_TOL
