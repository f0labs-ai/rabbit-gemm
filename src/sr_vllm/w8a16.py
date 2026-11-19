# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 F0 Labs

"""`sr_w8a16`: FP8 weights with bf16 activations for vLLM's linear layers, on rabbit-gemm's decode GEMVs.

Weights are vLLM's own online per-channel FP8 (`Fp8PtpcOnlineLinearMethod`: amax / 448 per output row),
held as a column-major [K, N] view of a row-major [N, K] tensor. Stock CUTLASS W8A8 reads the view and our
GEMVs read the row-major tensor underneath, so both run on the same bytes.

On the GEMV path activations are never quantized. That removes the per-token amax reduction that stock
FP8 fuses into every RMSNorm and SwiGLU kernel, and the activation rounding error with it.

A load-time autotune picks the kernel per (N, K) and decode bucket: our CUDA-core GEMV, our tensor-core
GEMV, or CUTLASS W8A8 including its activation quantization, each timed while streaming weights from HBM.
Inputs larger than the largest bucket (prefill) always take CUTLASS W8A8.
"""

import bisect
import enum
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from vllm import _custom_ops as ops
from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import QuantizationMethods, register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PtpcOnlineLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

import sr_vllm._C as _C  # noqa: PLR0402  # module import, so mypy applies ignore_missing_imports to the compiled extension
from sr_vllm import QUANTIZATION_NAME
from sr_vllm.archs import DEFAULT_ARCHS, SmArch, parse_archs, supporting

# Under the `vllm` namespace so vLLM's handler and level apply to the plan log.
logger = init_logger(f"vllm.{__name__}")

POLICY_ENV = "SR_VLLM_GEMV_POLICY"
# Synthetic weight copies rotated while timing total at least this multiple of L2, so every call streams
# its weights from HBM as it does across a model's distinct layers.
_L2_ROTATION = 4
_LAUNCHES_PER_GRAPH = 32
_GRAPH_REPLAYS = 8


class GemvPolicy(enum.Enum):
    """Which kernels the autotune may pick (`SR_VLLM_GEMV_POLICY`)."""

    AUTO = "auto"  # fastest measured candidate per (shape, bucket)
    SR = "sr"  # our GEMVs at every bucket they support; attribution arm
    CUTLASS = "cutlass"  # stock CUTLASS W8A8 everywhere, through this plugin; control arm


def policy_from_env() -> GemvPolicy:
    raw = os.environ.get(POLICY_ENV, GemvPolicy.AUTO.value)
    try:
        return GemvPolicy(raw)
    except ValueError:
        allowed = ", ".join(p.value for p in GemvPolicy)
        raise ValueError(f"{POLICY_ENV} must be one of {allowed}; got {raw!r}") from None


class KernelFamily(enum.Enum):
    SR_CUDA_CORE = "sr-cuda-core"
    SR_TENSOR_CORE = "sr-tensor-core"
    SR_TENSOR_CORE_WIDE = "sr-tensor-core-wide"
    CUTLASS_W8A8 = "cutlass-w8a8"


@dataclass(frozen=True)
class KernelChoice:
    family: KernelFamily
    variant: int = 0
    split: int = 1  # K-splits across CTAs; only the wide tensor-core kernel splits

    def __str__(self) -> str:
        if self.family is KernelFamily.CUTLASS_W8A8:
            return str(self.family.value)
        tail = f"/s{self.split}" if self.split > 1 else ""
        return f"{self.family.value}/v{self.variant}{tail}"


CUTLASS = KernelChoice(KernelFamily.CUTLASS_W8A8)


@dataclass(frozen=True)
class ShapePlan:
    """Kernel per decode bucket for one (N, K). An M runs the choice of the smallest bucket >= M."""

    buckets: tuple[int, ...]
    choices: tuple[KernelChoice, ...]

    def choose(self, m: int) -> KernelChoice:
        i = bisect.bisect_left(self.buckets, m)
        return self.choices[i] if i < len(self.choices) else CUTLASS


# Plans are per process: every vLLM worker autotunes its own weight shapes at load.
_PLANS: dict[tuple[int, int], ShapePlan] = {}


@dataclass(frozen=True)
class SplitWorkspace:
    """Split-K scratch for one (N, K): f32 partial tiles and the tile counters, which start at zero and are
    left at zero by every call. Allocated at load, never during CUDA-graph capture."""

    partials: torch.Tensor
    counters: torch.Tensor
    max_split: int


_WORKSPACES: dict[tuple[int, int], SplitWorkspace] = {}
# K-splits the autotune tries for the wide kernel; the kernel itself rejects splits past K / chunk.
_WIDE_SPLITS = (1, 2, 4, 8)


def _set_workspace(n: int, k: int, max_split: int, device: torch.device) -> SplitWorkspace:
    tiles = (n + int(_C.wide_min_channels_per_cta) - 1) // int(_C.wide_min_channels_per_cta)
    floats = max_split * int(_C.wide_max_m) * n if max_split > 1 else 0
    ws = SplitWorkspace(
        partials=torch.empty(floats, device=device, dtype=torch.float32),
        counters=torch.zeros(tiles, device=device, dtype=torch.int32),
        max_split=max_split,
    )
    _WORKSPACES[(n, k)] = ws
    return ws


def ensure_workspace(n: int, k: int, max_split: int, device: torch.device) -> SplitWorkspace:
    """The (N, K) workspace, grown if needed to hold `max_split` partial tiles of a 64-token batch."""
    have = _WORKSPACES.get((n, k))
    if have is not None and have.max_split >= max_split:
        return have
    return _set_workspace(n, k, max_split, device)


def plan_for(n: int, k: int) -> ShapePlan:
    try:
        return _PLANS[(n, k)]
    except KeyError:
        raise KeyError(
            f"{QUANTIZATION_NAME}: no kernel plan for N={n} K={k}; "
            "process_weights_after_loading never ran for a layer of this shape"
        ) from None


def run_choice(
    choice: KernelChoice,
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """y[M, N] = x[M, K] @ dequant(weight)[K, N] (+ bias); `weight` is the column-major [K, N] view."""
    if choice.family is KernelFamily.CUTLASS_W8A8:
        qx, x_scale = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        return ops.cutlass_scaled_mm(
            qx, weight, scale_a=x_scale, scale_b=weight_scale, out_dtype=x.dtype, bias=bias
        )
    if choice.family is KernelFamily.SR_TENSOR_CORE_WIDE:
        k, n = weight.shape
        ws = _WORKSPACES[(n, k)]  # ensure_workspace ran at load for every split the plan uses
        y = cast(
            torch.Tensor,
            _C.tc_wide_fp8w(
                x.contiguous(),
                weight.t(),
                weight_scale.view(-1),
                ws.partials,
                ws.counters,
                choice.split,
                choice.variant,
            ),
        )
    else:
        gemv = _C.gemv_fp8w if choice.family is KernelFamily.SR_CUDA_CORE else _C.tc_gemv_fp8w
        y = cast(torch.Tensor, gemv(x.contiguous(), weight.t(), weight_scale.view(-1), choice.variant))
    if bias is not None:
        y.add_(bias)
    return y


def _w8a16_linear(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    # Runs eagerly: once per size during CUDA-graph capture, and per call for uncaptured (prefill) sizes,
    # so the M dispatch below sees the real batch and never becomes a guard in the compiled graph.
    k, n = weight.shape
    return run_choice(plan_for(n, k).choose(x.shape[0]), x, weight, weight_scale, bias)


def _w8a16_linear_fake(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[1]))


# Opaque to torch.compile: Inductor sees only the fake implementation's shape.
_LIBRARY = torch.library.Library("sr_vllm", "FRAGMENT")  # type: ignore[no-untyped-call]
direct_register_custom_op(
    "w8a16_linear",
    _w8a16_linear,
    mutates_args=[],
    fake_impl=_w8a16_linear_fake,
    target_lib=_LIBRARY,
)


def _graph_us(launch: Callable[[int], object], copies: int) -> float:
    """Mean µs per call over CUDA-graph replays of back-to-back calls: GPU time without host launch cost."""
    side = torch.cuda.Stream()  # type: ignore[no-untyped-call]
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(copies):  # lazy module loading and allocator growth stay out of the graph
            launch(i)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        for i in range(_LAUNCHES_PER_GRAPH):
            launch(i % copies)
    graph.replay()
    start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
    start.record()
    for _ in range(_GRAPH_REPLAYS):
        graph.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) * 1e3 / (_GRAPH_REPLAYS * _LAUNCHES_PER_GRAPH)


class _CandidateTimer:
    """Times kernel choices for one (N, K) on synthetic FP8 weights rotated through more than L2."""

    def __init__(self, n: int, k: int, device: torch.device) -> None:
        self._k = k
        self._device = device
        l2_bytes = int(torch.cuda.get_device_properties(device).L2_cache_size)
        copies = max(2, math.ceil(_L2_ROTATION * l2_bytes / (n * k)))
        gen = torch.Generator(device=device).manual_seed(0)
        self._weights = [
            torch.randn(n, k, generator=gen, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn).t()
            for _ in range(copies)
        ]
        self._scales = [
            torch.rand(n, 1, generator=gen, device=device, dtype=torch.float32) * 1e-2 + 1e-3
            for _ in range(copies)
        ]

    def us_per_call(self, choice: KernelChoice, m: int) -> float | None:
        """GPU µs per call, or None when the kernel rejects this shape."""
        x = torch.randn(m, self._k, device=self._device, dtype=torch.bfloat16)

        def launch(i: int) -> torch.Tensor:
            return run_choice(choice, x, self._weights[i], self._scales[i], None)

        try:
            launch(0)
        except RuntimeError as err:  # the kernel's own shape validation (TORCH_CHECK in the binding)
            logger.debug("%s: %s rejects M=%d K=%d: %s", QUANTIZATION_NAME, choice, m, self._k, err)
            return None
        return _graph_us(launch, len(self._weights))


def _candidates(m: int, policy: GemvPolicy) -> list[KernelChoice]:
    found: list[KernelChoice] = []
    if m <= int(_C.max_m):
        found += [KernelChoice(KernelFamily.SR_CUDA_CORE, v) for v in range(int(_C.num_variants))]
    if m <= int(_C.tc_max_m):
        found += [KernelChoice(KernelFamily.SR_TENSOR_CORE, v) for v in range(int(_C.tc_num_variants))]
    # The wide kernel never beat CUTLASS W8A8 in the autotune, so it is timed only for policy sr, where it
    # is the fastest of ours at 24+ tokens on gate/up and down.
    if policy is GemvPolicy.SR and m <= int(_C.wide_max_m):
        found += [
            KernelChoice(KernelFamily.SR_TENSOR_CORE_WIDE, v, sp)
            for v in range(int(_C.wide_num_variants))
            for sp in _WIDE_SPLITS
        ]
    if policy is GemvPolicy.AUTO or not found:
        found.append(CUTLASS)
    return found


def build_plan(n: int, k: int, policy: GemvPolicy, buckets: Sequence[int], device: torch.device) -> ShapePlan:
    """Times every allowed candidate per bucket and keeps the fastest."""
    if policy is GemvPolicy.CUTLASS:
        return ShapePlan(tuple(buckets), tuple(CUTLASS for _ in buckets))
    timer = _CandidateTimer(n, k, device)
    _set_workspace(n, k, max(_WIDE_SPLITS), device)  # every split the timer may try
    choices: list[KernelChoice] = []
    report: list[str] = []
    for m in buckets:
        timed = [(us, c) for c in _candidates(m, policy) if (us := timer.us_per_call(c, m)) is not None]
        best_us, best = min(timed, key=lambda t: t[0], default=(math.nan, CUTLASS))
        cutlass_us = next((us for us, c in timed if c == CUTLASS), math.nan)
        choices.append(best)
        cutlass_txt = f"{cutlass_us:.2f}us" if math.isfinite(cutlass_us) else "not timed"
        report.append(f"M={m} {best} {best_us:.2f}us (cutlass {cutlass_txt})")
    del timer
    # Keep only the split-K scratch the chosen kernels use.
    _set_workspace(n, k, max([c.split for c in choices], default=1), device)
    torch.cuda.empty_cache()  # the synthetic weights must not shrink vLLM's KV-cache budget
    logger.info("%s plan N=%d K=%d policy=%s: %s", QUANTIZATION_NAME, n, k, policy.value, " | ".join(report))
    return ShapePlan(tuple(buckets), tuple(choices))


def decode_buckets() -> tuple[int, ...]:
    """vLLM's CUDA-graph batch sizes that our GEMVs can serve; powers of two when graphs are off."""
    limit = max(int(_C.max_m), int(_C.tc_max_m), int(_C.wide_max_m))
    sizes = get_current_vllm_config().compilation_config.cudagraph_capture_sizes or []
    buckets = sorted({s for s in sizes if s <= limit})
    if not buckets:
        buckets = [1 << i for i in range(limit.bit_length()) if 1 << i <= limit]
    return tuple(buckets)


class SrW8A16LinearMethod(Fp8PtpcOnlineLinearMethod):
    """vLLM's online per-channel FP8 weights; our GEMVs or CUTLASS W8A8 by measured plan."""

    def __init__(self, policy: GemvPolicy) -> None:
        super().__init__()  # type: ignore[no-untyped-call]  # reads the current vLLM config
        self._policy = policy
        self._buckets = decode_buckets()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        weight = cast(torch.Tensor, layer.weight)
        k, n = weight.shape
        # CUTLASS pads K and N to multiples of 16. Our GEMVs would write the padded width, so a padded
        # layer stays on the stock path.
        logical = (cast(int, layer.input_size_per_partition), cast(int, layer.output_size_per_partition))
        unpadded = (k, n) == logical
        layer.sr_w8a16_gemv = unpadded  # type: ignore[assignment]  # nn.Module attrs are typed as tensors
        if unpadded and (n, k) not in _PLANS:
            _PLANS[(n, k)] = build_plan(n, k, self._policy, self._buckets, weight.device)

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if envs.VLLM_BATCH_INVARIANT or not getattr(layer, "sr_w8a16_gemv", False):
            return super().apply(layer, x, bias)
        x2 = x.reshape(-1, x.shape[-1])
        y = cast(torch.Tensor, torch.ops.sr_vllm.w8a16_linear(x2, layer.weight, layer.weight_scale, bias))
        return y.view(*x.shape[:-1], y.shape[-1])


def built_archs() -> tuple[SmArch, ...]:
    """The SM architectures the compiled extension carries (baked in at build time)."""
    return parse_archs(str(_C.cuda_archs) or DEFAULT_ARCHS)


def require_supported_gpu() -> SmArch:
    """The built architecture this process's GPU runs. Raises when the binary has no code for it: without
    this check the first kernel launch fails with "no kernel image is available", after the weights loaded."""
    major, minor = torch.cuda.get_device_capability()
    arch = supporting(built_archs(), major, minor)
    if arch is None:
        built = ", ".join(f"sm_{a.name}" for a in built_archs())
        raise ValueError(
            f"{QUANTIZATION_NAME}: this GPU is sm_{major}{minor} ({torch.cuda.get_device_name()}), and the "
            f"plugin was built for {built}. Rebuild it with SR_CUDA_ARCH including this architecture, or "
            "serve with --quantization fp8."
        )
    return arch


class SrW8A16Config(QuantizationConfig):
    """Online: quantizes a bf16 checkpoint at load, so there is no checkpoint config file to read."""

    def __init__(self) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.policy = policy_from_env()
        arch = require_supported_gpu()
        logger.info(
            "%s: kernels built for sm_%s on %s", QUANTIZATION_NAME, arch.name, torch.cuda.get_device_name()
        )

    def get_name(self) -> QuantizationMethods:
        return cast(QuantizationMethods, QUANTIZATION_NAME)

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16]  # the GEMVs read bf16 activations only

    @classmethod
    def get_min_capability(cls) -> int:
        # The lowest architecture the binary carries; require_supported_gpu checks the exact match.
        return min(a.capability for a in built_archs())

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SrW8A16Config":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            return SrW8A16LinearMethod(self.policy)
        return None  # embeddings, LM head and attention stay unquantized, as with stock online FP8


def register_quantization() -> None:
    register_quantization_config(QUANTIZATION_NAME)(SrW8A16Config)
