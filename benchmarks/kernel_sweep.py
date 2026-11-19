# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 F0 Labs

"""Kernel microbenchmark: decode-shape GEMMs at small M, every kernel family, one GPU.

y[M, N] = x[M, K] · W[N, K]^T with BF16 activations. Families:
  bf16      cuBLAS BF16 through torch.nn.functional.linear: the unquantized baseline
  w8a8      vLLM's CUTLASS FP8 scaled_mm: per-token FP8 activations, per-channel FP8 weights; x quantized
            beforehand
  w8a8+q    the same with vLLM's dynamic per-token activation quantization inside the timed call
  marlin    vLLM's FP8 Marlin: per-channel FP8 weights, BF16 activations (weight-only)
  sr-cc     our CUDA-core W8A16 GEMV (M <= 8), best of its tile variants
  sr-tc     our tensor-core W8A16 GEMV (M <= 64), best of its variants
  sr-wide   our wide tensor-core kernel (kept as a negative result), best of variants x splits, M >= 16

Every FP8 family reads the same FP8 weights and per-channel scales. Weights rotate through copies totalling at
least 4x L2, and each CUDA graph cycles through all of them, so every call streams its weights from DRAM as it
does across a model's distinct layers. A trial is REPLAYS replays of one graph. Variants are screened with one
trial each; each family's best then runs TRIALS trials, interleaved across families so clock drift hits them
alike. One JSON object per line on stdout.

Run it where the plugin is installed, e.g.:

    python benchmarks/kernel_sweep.py --models qwen3-8b --trials 5 > sweep.jsonl
"""

import argparse
import json
import math
import time
from collections.abc import Callable

import torch
import vllm._custom_ops as ops
from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as marlin

from sr_vllm import _C as C  # after torch, whose libraries the extension links against

SHAPES: dict[str, list[tuple[str, int, int]]] = {
    "qwen3-8b": [("qkv", 6144, 4096), ("o", 4096, 4096), ("gate_up", 24576, 4096), ("down", 4096, 12288)],
    "qwen3.8-27b": [
        ("deltanet_qkvz", 16384, 5120),
        ("deltanet_ba", 96, 5120),
        ("mixer_out", 5120, 6144),
        ("attn_qkv", 14336, 5120),
        ("gate_up", 34816, 5120),
        ("down", 5120, 17408),
    ],
    "llama-3.3-70b": [
        ("qkv", 10240, 8192),
        ("o", 8192, 8192),
        ("gate_up", 57344, 8192),
        ("down", 8192, 28672),
    ],
}
MS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
MIN_LAUNCHES = 32
REPLAYS = 8
L2_ROTATION = 4
FP8_MAX = 448.0

Launch = Callable[[int], torch.Tensor]


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


class Timed:
    """One CUDA graph of back-to-back calls cycling through every weight copy; µs per call per trial."""

    def __init__(self, launch: Launch, copies: int) -> None:
        self.launches = max(MIN_LAUNCHES, copies)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for i in range(min(copies, 4)):  # lazy module loading and allocator growth stay out of the graph
                launch(i)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            for i in range(self.launches):
                launch(i % copies)
        self.graph.replay()
        torch.cuda.synchronize()

    def trial_us(self) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(REPLAYS):
            self.graph.replay()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end)) * 1e3 / (REPLAYS * self.launches)


def rel_err(y: torch.Tensor, ref: torch.Tensor) -> float:
    return float((y.float() - ref).norm() / ref.norm().clamp_min(1e-30))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(SHAPES))
    p.add_argument("--ms", default=",".join(str(m) for m in MS))
    p.add_argument("--trials", type=int, default=9)
    a = p.parse_args()

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    l2 = int(props.L2_cache_size)
    gpu = torch.cuda.get_device_name()
    emit(
        {
            "kind": "meta",
            "gpu": gpu,
            "capability": list(torch.cuda.get_device_capability()),
            "sms": props.multi_processor_count,
            "l2_bytes": l2,
            "torch": torch.__version__,
            "cuda_archs": str(C.cuda_archs),
            "time": time.time(),
            "replays": REPLAYS,
            "min_launches": MIN_LAUNCHES,
            "l2_rotation": L2_ROTATION,
            "trials": a.trials,
        }
    )
    ms = [int(v) for v in a.ms.split(",")]
    gen = torch.Generator(device=dev).manual_seed(0)

    for model in a.models.split(","):
        for name, n, k in SHAPES[model]:
            t_shape = time.time()
            w_bf16 = (torch.randn(n, k, generator=gen, device=dev, dtype=torch.float32) / math.sqrt(k)).to(
                torch.bfloat16
            )
            amax = w_bf16.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
            scale = (amax / FP8_MAX).float()  # [N, 1], per output channel
            # [N, K] row-major
            w_fp8 = (w_bf16.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            copies8 = max(2, math.ceil(L2_ROTATION * l2 / (n * k)))
            copies16 = max(2, math.ceil(L2_ROTATION * l2 / (2 * n * k)))
            w8 = [w_fp8.clone() for _ in range(copies8)]
            s8 = [scale.clone() for _ in range(copies8)]
            w16 = [w_bf16.clone() for _ in range(copies16)]
            mw: list[torch.Tensor] = []
            ms_: list[torch.Tensor] = []
            marlin_error = ""
            try:
                for i in range(copies8):
                    layer = torch.nn.Module()
                    layer.output_size_per_partition = n
                    layer.input_size_per_partition = k
                    layer.orig_dtype = torch.bfloat16
                    # [K, N] view, as vLLM holds it
                    layer.weight = torch.nn.Parameter(w8[i].t(), requires_grad=False)
                    layer.weight_scale = torch.nn.Parameter(s8[i].view(-1).clone(), requires_grad=False)
                    marlin.prepare_fp8_layer_for_marlin(layer, size_k_first=True)
                    mw.append(layer.weight.data)
                    ms_.append(layer.weight_scale.data)
            except Exception as err:  # recorded; the family is skipped
                marlin_error = f"{type(err).__name__}: {err}"[:300]
                mw, ms_ = [], []
            ref_w8 = (w_fp8.float() * scale).t().contiguous()  # dequantized FP8 weights, [K, N]
            ref_w16 = w_bf16.float().t().contiguous()
            emit(
                {
                    "kind": "shape",
                    "model": model,
                    "shape": name,
                    "n": n,
                    "k": k,
                    "fp8_bytes": n * k,
                    "copies_fp8": copies8,
                    "copies_bf16": copies16,
                    "marlin_error": marlin_error,
                }
            )

            for m in ms:
                x = torch.randn(m, k, generator=gen, device=dev, dtype=torch.bfloat16)
                qx, xs = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
                ref8 = x.float() @ ref_w8
                ref16 = x.float() @ ref_w16
                base = {"model": model, "shape": name, "n": n, "k": k, "m": m}

                families: dict[str, tuple[Launch, int]] = {
                    "bf16": (lambda i: torch.nn.functional.linear(x, w16[i]), copies16),
                    "w8a8": (
                        lambda i: ops.cutlass_scaled_mm(
                            qx, w8[i].t(), scale_a=xs, scale_b=s8[i], out_dtype=torch.bfloat16
                        ),
                        copies8,
                    ),
                    "w8a8+q": (lambda i: _w8a8_with_quant(x, w8[i], s8[i]), copies8),
                }
                if mw:
                    families["marlin"] = (
                        lambda i: marlin.apply_fp8_marlin_linear(
                            input=x,
                            weight=mw[i],
                            weight_scale=ms_[i],
                            workspace=None,
                            size_n=n,
                            size_k=k,
                            bias=None,
                        ),
                        copies8,
                    )

                # Our families: screen every variant once, keep the best per family.
                candidates: dict[str, list[tuple[str, Launch]]] = {"sr-cc": [], "sr-tc": [], "sr-wide": []}
                if m <= int(C.max_m):
                    for v in range(int(C.num_variants)):
                        candidates["sr-cc"].append((f"v{v}", _bind(C.gemv_fp8w, x, w8, s8, v)))
                if m <= int(C.tc_max_m):
                    for v in range(int(C.tc_num_variants)):
                        candidates["sr-tc"].append((f"v{v}", _bind(C.tc_gemv_fp8w, x, w8, s8, v)))
                if 16 <= m <= int(C.wide_max_m):
                    tiles = (n + int(C.wide_min_channels_per_cta) - 1) // int(C.wide_min_channels_per_cta)
                    max_split = int(C.wide_max_split)
                    ws = torch.empty(max_split * int(C.wide_max_m) * n, device=dev, dtype=torch.float32)
                    counters = torch.zeros(tiles, device=dev, dtype=torch.int32)
                    for v in range(int(C.wide_num_variants)):
                        for sp in (1, 2, 4, 8):
                            if sp > max_split:
                                continue
                            candidates["sr-wide"].append(
                                (f"v{v}/s{sp}", _bind_wide(C.tc_wide_fp8w, x, w8, s8, ws, counters, sp, v))
                            )
                for fam, cands in candidates.items():
                    best: tuple[float, str, Launch] | None = None
                    for var, launch in cands:
                        try:
                            us = Timed(launch, copies8).trial_us()
                        except RuntimeError as err:
                            emit(
                                {
                                    "kind": "reject",
                                    **base,
                                    "kernel": fam,
                                    "variant": var,
                                    "error": str(err)[:200],
                                }
                            )
                            continue
                        emit({"kind": "screen", **base, "kernel": fam, "variant": var, "us": round(us, 3)})
                        if best is None or us < best[0]:
                            best = (us, var, launch)
                    if best is not None:
                        families[fam] = (best[2], copies8)
                        emit({"kind": "best", **base, "kernel": fam, "variant": best[1]})

                timed: dict[str, Timed] = {}
                for fam, (launch, copies) in families.items():
                    try:
                        timed[fam] = Timed(launch, copies)
                        y = launch(0)
                        torch.cuda.synchronize()
                        ref = ref16 if fam == "bf16" else ref8
                        emit(
                            {
                                "kind": "acc",
                                **base,
                                "kernel": fam,
                                "rel_err": rel_err(y, ref),
                                "rel_err_vs_bf16_weights": rel_err(y, ref16),
                            }
                        )
                    except Exception as err:  # recorded; the family is skipped at this M
                        emit(
                            {
                                "kind": "error",
                                **base,
                                "kernel": fam,
                                "error": f"{type(err).__name__}: {err}"[:300],
                            }
                        )
                t0 = time.time()
                for trial in range(a.trials):
                    for fam, t in timed.items():
                        emit(
                            {
                                "kind": "time",
                                **base,
                                "kernel": fam,
                                "trial": trial,
                                "us": round(t.trial_us(), 4),
                            }
                        )
                emit({"kind": "block", **base, "t_start": t0, "t_end": time.time()})
                del timed, families, candidates
                torch.cuda.empty_cache()
            # Release this shape's weights before the next shape allocates its own.
            w8, s8, w16, mw, ms_ = [], [], [], [], []
            del ref_w8, ref_w16
            torch.cuda.empty_cache()
            emit(
                {
                    "kind": "shape_done",
                    "model": model,
                    "shape": name,
                    "seconds": round(time.time() - t_shape, 1),
                }
            )


def _w8a8_with_quant(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """The unfused W8A8 call: dynamic per-token FP8 quantization of x, then CUTLASS scaled_mm."""
    qx, xs = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
    return ops.cutlass_scaled_mm(qx, w.t(), scale_a=xs, scale_b=s, out_dtype=torch.bfloat16)


def _bind(
    fn: Callable[..., torch.Tensor], x: torch.Tensor, w8: list[torch.Tensor], s8: list[torch.Tensor], v: int
) -> Launch:
    return lambda i: fn(x, w8[i], s8[i].view(-1), v)


def _bind_wide(
    fn: Callable[..., torch.Tensor],
    x: torch.Tensor,
    w8: list[torch.Tensor],
    s8: list[torch.Tensor],
    ws: torch.Tensor,
    counters: torch.Tensor,
    split: int,
    v: int,
) -> Launch:
    return lambda i: fn(x, w8[i], s8[i].view(-1), ws, counters, split, v)


if __name__ == "__main__":
    main()
