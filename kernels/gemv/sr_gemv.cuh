// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 F0 Labs

// Decode-time linear layers: y[M, N] = x[M, K] · W[N, K]^T for a small batch of tokens (M ≤ 8).
//
// Roofline: at M ≤ 8 the arithmetic intensity is ≈ 2M / bytes-per-weight FLOP/B (BF16: M, FP8: 2M),
// far below H100's ridge (295 FLOP/B BF16), so the kernel's time is weight bytes / HBM bandwidth.
// Budget: one call per linear layer per decode step; e.g. a 4096×4096 BF16 projection is 32 MiB,
// 10.0 µs at 3.35 TB/s peak, ~11 µs at the ~90% bandwidth a streaming kernel can sustain.
//
// Design:
//  * One warp owns ROWS output rows. Its 32 lanes walk each row in 16-byte chunks and issue U chunks
//    per row before consuming any, so every SM keeps tens of KB in flight. Little's law: ~2.3 MB
//    outstanding saturates H100 HBM3 (3.35 TB/s × ~700 ns).
//  * Weights are read exactly once per step: streaming (evict-first) loads keep them from evicting
//    the activations, which every warp reuses and which therefore come from L1 via the read-only path.
//  * FP32 accumulation; one warp-shuffle reduction and one store per output.
//  * FP8 (E4M3) weights carry one FP32 scale per output row and are expanded to FP32 in registers
//    with the hardware cvt (e4m3x2 → f16x2). Activations stay BF16 (W8A16), so no per-token
//    activation quantization kernel is needed.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "sr_launch.cuh"

namespace sr::gemv {

inline constexpr int kWarpsPerBlock = 8;
inline constexpr int kThreads = kWarpsPerBlock * 32;
inline constexpr int kMaxM = 8;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
  return v;
}

template <int M, int ROWS, int U>
__global__ void __launch_bounds__(kThreads)
    gemv_bf16_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                     __nv_bfloat16* __restrict__ y, int n, int k) {
  SR_TRACE_CTA(sr::trace::kGemvCudaCoreBf16);
  sr::pdl_trigger();
  const int lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5)) * ROWS;
  if (row0 >= n) {
    sr::pdl_wait();
    return;
  }
  const int chunks = k >> 3;  // 8 BF16 per 16-byte chunk
  const uint4* w4 = reinterpret_cast<const uint4*>(w);
  const uint4* x4 = reinterpret_cast<const uint4*>(x);

  float acc[ROWS][M];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;

  for (int base = lane; base < chunks; base += 32 * U) {
    uint4 wv[U][ROWS];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int c = base + 32 * u;
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const int row = min(row0 + r, n - 1);
        wv[u][r] = c < chunks ? __ldcs(w4 + static_cast<size_t>(row) * chunks + c) : make_uint4(0, 0, 0, 0);
      }
    }
    // Weights never depend on the previous kernel: they stream before the wait, x only after it.
    sr::pdl_wait();
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int c = base + 32 * u;
      if (c >= chunks) break;
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const uint4 xv = __ldg(x4 + static_cast<size_t>(m) * chunks + c);
        const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 xf = __bfloat1622float2(xp[i]);
#pragma unroll
          for (int r = 0; r < ROWS; ++r) {
            const float2 wf = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(&wv[u][r])[i]);
            acc[r][m] = fmaf(wf.x, xf.x, acc[r][m]);
            acc[r][m] = fmaf(wf.y, xf.y, acc[r][m]);
          }
        }
      }
    }
  }

  sr::pdl_wait();  // lanes with no chunks still order their stores after the predecessor
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = warp_sum(acc[r][m]);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const int row = row0 + r;
      if (row < n) {
#pragma unroll
        for (int m = 0; m < M; ++m) y[static_cast<size_t>(m) * n + row] = __float2bfloat16(acc[r][m]);
      }
    }
  }
}

template <int M, int ROWS, int U>
__global__ void __launch_bounds__(kThreads)
    gemv_fp8w_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w,
                     const float* __restrict__ scale, __nv_bfloat16* __restrict__ y, int n, int k) {
  SR_TRACE_CTA(sr::trace::kGemvCudaCoreFp8);
  sr::pdl_trigger();
  const int lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5)) * ROWS;
  if (row0 >= n) {
    sr::pdl_wait();
    return;
  }
  const int chunks = k >> 4;  // 16 FP8 per 16-byte chunk
  const int x_chunks = k >> 3;
  const uint4* w4 = reinterpret_cast<const uint4*>(w);
  const uint4* x4 = reinterpret_cast<const uint4*>(x);

  float acc[ROWS][M];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;

  for (int base = lane; base < chunks; base += 32 * U) {
    uint4 wv[U][ROWS];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int c = base + 32 * u;
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const int row = min(row0 + r, n - 1);
        wv[u][r] = c < chunks ? __ldcs(w4 + static_cast<size_t>(row) * chunks + c) : make_uint4(0, 0, 0, 0);
      }
    }
    // Weights never depend on the previous kernel: they stream before the wait, x only after it.
    sr::pdl_wait();
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int c = base + 32 * u;
      if (c >= chunks) break;
      // Expand each row's 16 E4M3 weights to FP32 once; reuse them for every token m.
      float wf[ROWS][16];
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const __nv_fp8x2_storage_t* wp = reinterpret_cast<const __nv_fp8x2_storage_t*>(&wv[u][r]);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const __half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(wp[i], __NV_E4M3);
          const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2));
          wf[r][2 * i] = f.x;
          wf[r][2 * i + 1] = f.y;
        }
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const uint4 xa = __ldg(x4 + static_cast<size_t>(m) * x_chunks + 2 * c);
        const uint4 xb = __ldg(x4 + static_cast<size_t>(m) * x_chunks + 2 * c + 1);
        const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&xa);
        const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&xb);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 lo = __bfloat1622float2(pa[i]);
          const float2 hi = __bfloat1622float2(pb[i]);
#pragma unroll
          for (int r = 0; r < ROWS; ++r) {
            acc[r][m] = fmaf(wf[r][2 * i], lo.x, acc[r][m]);
            acc[r][m] = fmaf(wf[r][2 * i + 1], lo.y, acc[r][m]);
            acc[r][m] = fmaf(wf[r][8 + 2 * i], hi.x, acc[r][m]);
            acc[r][m] = fmaf(wf[r][8 + 2 * i + 1], hi.y, acc[r][m]);
          }
        }
      }
    }
  }

  sr::pdl_wait();  // lanes with no chunks still order their stores after the predecessor
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = warp_sum(acc[r][m]);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const int row = row0 + r;
      if (row < n) {
        const float s = scale[row];
#pragma unroll
        for (int m = 0; m < M; ++m) y[static_cast<size_t>(m) * n + row] = __float2bfloat16(acc[r][m] * s);
      }
    }
  }
}

// Tile configurations (rows per warp, chunks in flight per row) selected by index; the benchmark
// sweeps them, production dispatch picks per (shape, M, GPU) from autotuning results.
inline constexpr int kNumVariants = 5;

template <int M, int ROWS, int U>
cudaError_t launch_bf16(const void* x, const void* w, void* y, int n, int k, cudaStream_t stream) {
  const dim3 grid((n + kWarpsPerBlock * ROWS - 1) / (kWarpsPerBlock * ROWS));
  return sr::launch_pdl(gemv_bf16_kernel<M, ROWS, U>, grid, dim3(kThreads), 0, stream,
                        static_cast<const __nv_bfloat16*>(x), static_cast<const __nv_bfloat16*>(w),
                        static_cast<__nv_bfloat16*>(y), n, k);
}

template <int M, int ROWS, int U>
cudaError_t launch_fp8w(const void* x, const void* w, const float* scale, void* y, int n, int k, cudaStream_t stream) {
  const dim3 grid((n + kWarpsPerBlock * ROWS - 1) / (kWarpsPerBlock * ROWS));
  return sr::launch_pdl(gemv_fp8w_kernel<M, ROWS, U>, grid, dim3(kThreads), 0, stream,
                        static_cast<const __nv_bfloat16*>(x), static_cast<const uint8_t*>(w), scale,
                        static_cast<__nv_bfloat16*>(y), n, k);
}

template <int M>
cudaError_t dispatch_bf16_m(const void* x, const void* w, void* y, int n, int k, int variant, cudaStream_t s) {
  switch (variant) {
    case 0: return launch_bf16<M, 1, 4>(x, w, y, n, k, s);
    case 1: return launch_bf16<M, 2, 2>(x, w, y, n, k, s);
    case 2: return launch_bf16<M, 1, 8>(x, w, y, n, k, s);
    case 3: return launch_bf16<M, 2, 4>(x, w, y, n, k, s);
    case 4: return launch_bf16<M, 4, 2>(x, w, y, n, k, s);
    default: return cudaErrorInvalidValue;
  }
}

template <int M>
cudaError_t dispatch_fp8w_m(const void* x, const void* w, const float* scale, void* y, int n, int k, int variant,
                            cudaStream_t s) {
  switch (variant) {
    case 0: return launch_fp8w<M, 1, 4>(x, w, scale, y, n, k, s);
    case 1: return launch_fp8w<M, 2, 2>(x, w, scale, y, n, k, s);
    case 2: return launch_fp8w<M, 1, 8>(x, w, scale, y, n, k, s);
    case 3: return launch_fp8w<M, 2, 4>(x, w, scale, y, n, k, s);
    case 4: return launch_fp8w<M, 4, 2>(x, w, scale, y, n, k, s);
    default: return cudaErrorInvalidValue;
  }
}

// Requirements: 1 ≤ m ≤ 8; k a multiple of 16 (FP8) or 8 (BF16); x, w, y 16-byte aligned, row-major.
inline cudaError_t gemv_bf16(const void* x, const void* w, void* y, int m, int n, int k, int variant,
                             cudaStream_t s) {
  if (k % 8 != 0 || n < 1) return cudaErrorInvalidValue;
  switch (m) {
    case 1: return dispatch_bf16_m<1>(x, w, y, n, k, variant, s);
    case 2: return dispatch_bf16_m<2>(x, w, y, n, k, variant, s);
    case 3: return dispatch_bf16_m<3>(x, w, y, n, k, variant, s);
    case 4: return dispatch_bf16_m<4>(x, w, y, n, k, variant, s);
    case 5: return dispatch_bf16_m<5>(x, w, y, n, k, variant, s);
    case 6: return dispatch_bf16_m<6>(x, w, y, n, k, variant, s);
    case 7: return dispatch_bf16_m<7>(x, w, y, n, k, variant, s);
    case 8: return dispatch_bf16_m<8>(x, w, y, n, k, variant, s);
    default: return cudaErrorInvalidValue;
  }
}

inline cudaError_t gemv_fp8w(const void* x, const void* w, const float* scale, void* y, int m, int n, int k,
                             int variant, cudaStream_t s) {
  if (k % 16 != 0 || n < 1) return cudaErrorInvalidValue;
  switch (m) {
    case 1: return dispatch_fp8w_m<1>(x, w, scale, y, n, k, variant, s);
    case 2: return dispatch_fp8w_m<2>(x, w, scale, y, n, k, variant, s);
    case 3: return dispatch_fp8w_m<3>(x, w, scale, y, n, k, variant, s);
    case 4: return dispatch_fp8w_m<4>(x, w, scale, y, n, k, variant, s);
    case 5: return dispatch_fp8w_m<5>(x, w, scale, y, n, k, variant, s);
    case 6: return dispatch_fp8w_m<6>(x, w, scale, y, n, k, variant, s);
    case 7: return dispatch_fp8w_m<7>(x, w, scale, y, n, k, variant, s);
    case 8: return dispatch_fp8w_m<8>(x, w, scale, y, n, k, variant, s);
    default: return cudaErrorInvalidValue;
  }
}

}  // namespace sr::gemv
