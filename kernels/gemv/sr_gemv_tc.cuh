// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 F0 Labs

// Tensor-core skinny GEMM for decode: y[M, N] = x[M, K] · W[N, K]^T with M ≤ 64 tokens.
//
// v1 (sr_gemv.cuh) runs on FP32 FMA units: at M = 8 with FP8 weights it needs ~16 FLOP per weight
// byte — at 3 TB/s that is ~48 TFLOP/s of FP32 plus the e4m3→f16 conversions, past what the CUDA cores
// sustain, so it stops being bandwidth-bound. This version keeps the memory pattern and moves the
// math to mma.sync m16n8k16:
//
//  * "Swap AB": the MMA's 16-row A operand is a 16-channel slice of W, its 8-column B operand is up to
//    8 tokens of x, so C = y^T tile [16 channels × 8 tokens]. One A fragment serves every token tile.
//  * K permutation: the MMA sums over K, so any permutation applied to both operands is exact. We map
//    lane (g = lane/4, t = lane%4)'s fragment slots {2t, 2t+1, 2t+8, 2t+9} of k-step s to physical
//    k = base + 8t + 4s + {0,1,2,3}, so each lane's operands for two k-steps are ONE contiguous
//    16-byte load per row (BF16) — four lanes read a row's 64 contiguous bytes, fully coalesced.
//    FP8 rows pack four k-steps per 16-byte load.
//  * Split-K: the WARPS_K warps of a block each own a K slice of the same 16 channels; partial C tiles
//    are reduced through shared memory. A 4096-channel layer is then 256 blocks × WARPS_K warps —
//    enough loads in flight per SM to saturate HBM (Little's law).
//  * FP8 (E4M3) weights are expanded with the hardware cvt (e4m3x2 → f16x2) and multiplied in f16 with
//    f32 accumulation; activations are converted bf16 → f16 in registers. Activations above f16 range
//    (|x| > 65504) would overflow: callers keep x post-normalization (true for every projection input).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "sr_launch.cuh"

namespace sr::gemv_tc {

__device__ __forceinline__ void mma_bf16(float c[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_f16(float c[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t pair) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair & 0xffffu), __NV_E4M3);
  return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.y) << 16);
}

__device__ __forceinline__ uint32_t bf16x2_to_f16x2(uint32_t v) {
  const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&v);
  const __half2 h = __float22half2_rn(__bfloat1622float2(b));
  return *reinterpret_cast<const uint32_t*>(&h);
}

// Shared epilogue: reduce WARPS_K partial tiles, scale, store y[token][channel].
template <int TOKEN_TILES, int WARPS_K>
__device__ __forceinline__ void reduce_and_store(float (&c)[TOKEN_TILES][4], __nv_bfloat16* __restrict__ y,
                                                 const float* __restrict__ scale, int m, int n, int n0) {
  __shared__ float partial[WARPS_K][TOKEN_TILES][4][32];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int tt = 0; tt < TOKEN_TILES; ++tt)
#pragma unroll
    for (int i = 0; i < 4; ++i) partial[warp][tt][i][lane] = c[tt][i];
  __syncthreads();
  if (warp != 0) return;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int row_lo = n0 + g;
  const int row_hi = n0 + g + 8;
  const float s_lo = scale != nullptr && row_lo < n ? scale[row_lo] : 1.f;
  const float s_hi = scale != nullptr && row_hi < n ? scale[row_hi] : 1.f;
#pragma unroll
  for (int tt = 0; tt < TOKEN_TILES; ++tt) {
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float acc = 0.f;
#pragma unroll
      for (int w = 0; w < WARPS_K; ++w) acc += partial[w][tt][i][lane];
      v[i] = acc;
    }
    // C fragment: c0,c1 -> (channel g, tokens 2t, 2t+1); c2,c3 -> (channel g+8, tokens 2t, 2t+1).
    const int tok = tt * 8 + 2 * t;
    if (row_lo < n) {
      if (tok < m) y[static_cast<size_t>(tok) * n + row_lo] = __float2bfloat16(v[0] * s_lo);
      if (tok + 1 < m) y[static_cast<size_t>(tok + 1) * n + row_lo] = __float2bfloat16(v[1] * s_lo);
    }
    if (row_hi < n) {
      if (tok < m) y[static_cast<size_t>(tok) * n + row_hi] = __float2bfloat16(v[2] * s_hi);
      if (tok + 1 < m) y[static_cast<size_t>(tok + 1) * n + row_hi] = __float2bfloat16(v[3] * s_hi);
    }
  }
}

template <int TOKEN_TILES, int WARPS_K, int U>
__global__ void __launch_bounds__(WARPS_K * 32)
    tc_gemv_bf16_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                        __nv_bfloat16* __restrict__ y, int m, int n, int k) {
  SR_TRACE_CTA(sr::trace::kGemvTensorCoreBf16);
  sr::pdl_trigger();
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int n0 = blockIdx.x * 16;
  const int kblocks = k >> 5;  // 32 physical k per block = two k-steps
  const int per_warp = (kblocks + WARPS_K - 1) / WARPS_K;
  const int kb_begin = warp * per_warp;
  const int kb_end = min(kblocks, kb_begin + per_warp);
  const uint4* wa = reinterpret_cast<const uint4*>(w + static_cast<size_t>(min(n0 + g, n - 1)) * k) + t;
  const uint4* wb = reinterpret_cast<const uint4*>(w + static_cast<size_t>(min(n0 + g + 8, n - 1)) * k) + t;

  float c[TOKEN_TILES][4];
#pragma unroll
  for (int tt = 0; tt < TOKEN_TILES; ++tt)
#pragma unroll
    for (int i = 0; i < 4; ++i) c[tt][i] = 0.f;

  for (int kb = kb_begin; kb < kb_end; kb += U) {
    uint4 lo[U], hi[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const bool ok = kb + u < kb_end;
      lo[u] = ok ? __ldcs(wa + static_cast<size_t>(kb + u) * 4) : make_uint4(0, 0, 0, 0);
      hi[u] = ok ? __ldcs(wb + static_cast<size_t>(kb + u) * 4) : make_uint4(0, 0, 0, 0);
    }
    sr::pdl_wait();  // weights streamed before the wait; activations only after it
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (kb + u >= kb_end) break;
#pragma unroll
      for (int tt = 0; tt < TOKEN_TILES; ++tt) {
        const int token = tt * 8 + g;
        const uint4 xv = token < m
                             ? __ldg(reinterpret_cast<const uint4*>(x + static_cast<size_t>(token) * k) +
                                     static_cast<size_t>(kb + u) * 4 + t)
                             : make_uint4(0, 0, 0, 0);
        const uint32_t a0[4] = {lo[u].x, hi[u].x, lo[u].y, hi[u].y};
        const uint32_t b0[2] = {xv.x, xv.y};
        mma_bf16(c[tt], a0, b0);
        const uint32_t a1[4] = {lo[u].z, hi[u].z, lo[u].w, hi[u].w};
        const uint32_t b1[2] = {xv.z, xv.w};
        mma_bf16(c[tt], a1, b1);
      }
    }
  }
  sr::pdl_wait();  // warps with an empty K range still order their stores after the predecessor
  reduce_and_store<TOKEN_TILES, WARPS_K>(c, y, nullptr, m, n, n0);
}

template <int TOKEN_TILES, int WARPS_K, int U>
__global__ void __launch_bounds__(WARPS_K * 32)
    tc_gemv_fp8w_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w,
                        const float* __restrict__ scale, __nv_bfloat16* __restrict__ y, int m, int n, int k) {
  SR_TRACE_CTA(sr::trace::kGemvTensorCoreFp8);
  sr::pdl_trigger();
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int n0 = blockIdx.x * 16;
  const int kblocks = k >> 6;  // 64 physical k per block = four k-steps
  const int per_warp = (kblocks + WARPS_K - 1) / WARPS_K;
  const int kb_begin = warp * per_warp;
  const int kb_end = min(kblocks, kb_begin + per_warp);
  const uint4* wa = reinterpret_cast<const uint4*>(w + static_cast<size_t>(min(n0 + g, n - 1)) * k) + t;
  const uint4* wb = reinterpret_cast<const uint4*>(w + static_cast<size_t>(min(n0 + g + 8, n - 1)) * k) + t;

  float c[TOKEN_TILES][4];
#pragma unroll
  for (int tt = 0; tt < TOKEN_TILES; ++tt)
#pragma unroll
    for (int i = 0; i < 4; ++i) c[tt][i] = 0.f;

  for (int kb = kb_begin; kb < kb_end; kb += U) {
    uint4 lo[U], hi[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const bool ok = kb + u < kb_end;
      lo[u] = ok ? __ldcs(wa + static_cast<size_t>(kb + u) * 4) : make_uint4(0, 0, 0, 0);
      hi[u] = ok ? __ldcs(wb + static_cast<size_t>(kb + u) * 4) : make_uint4(0, 0, 0, 0);
    }
    sr::pdl_wait();  // weights streamed before the wait; activations only after it
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (kb + u >= kb_end) break;
      const uint32_t wlo[4] = {lo[u].x, lo[u].y, lo[u].z, lo[u].w};
      const uint32_t whi[4] = {hi[u].x, hi[u].y, hi[u].z, hi[u].w};
      // Expand the weight fragments once per k-step; every token tile reuses them.
      uint32_t a[4][4];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        a[s][0] = e4m3x2_to_f16x2(wlo[s]);
        a[s][1] = e4m3x2_to_f16x2(whi[s]);
        a[s][2] = e4m3x2_to_f16x2(wlo[s] >> 16);
        a[s][3] = e4m3x2_to_f16x2(whi[s] >> 16);
      }
#pragma unroll
      for (int tt = 0; tt < TOKEN_TILES; ++tt) {
        const int token = tt * 8 + g;
        uint4 xa = make_uint4(0, 0, 0, 0);
        uint4 xb = make_uint4(0, 0, 0, 0);
        if (token < m) {
          const uint4* xp = reinterpret_cast<const uint4*>(x + static_cast<size_t>(token) * k) +
                            static_cast<size_t>(kb + u) * 8 + 2 * t;
          xa = __ldg(xp);
          xb = __ldg(xp + 1);
        }
        const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const uint32_t b[2] = {bf16x2_to_f16x2(xw[2 * s]), bf16x2_to_f16x2(xw[2 * s + 1])};
          mma_f16(c[tt], a[s], b);
        }
      }
    }
  }
  sr::pdl_wait();  // warps with an empty K range still order their stores after the predecessor
  reduce_and_store<TOKEN_TILES, WARPS_K>(c, y, scale, m, n, n0);
}

inline constexpr int kNumVariants = 4;  // (WARPS_K, U) in {(4,2), (4,4), (8,2), (8,4)}
inline constexpr int kMaxM = 64;        // TOKEN_TILES up to 8 tiles of 8 tokens

template <int TT>
cudaError_t dispatch_bf16_tt(const void* x, const void* w, void* y, int m, int n, int k, int variant,
                             cudaStream_t s) {
  const dim3 grid((n + 15) / 16);
  const auto* xp = static_cast<const __nv_bfloat16*>(x);
  const auto* wp = static_cast<const __nv_bfloat16*>(w);
  auto* yp = static_cast<__nv_bfloat16*>(y);
  switch (variant) {
    case 0: return sr::launch_pdl(tc_gemv_bf16_kernel<TT, 4, 2>, grid, dim3(128), 0, s, xp, wp, yp, m, n, k);
    case 1: return sr::launch_pdl(tc_gemv_bf16_kernel<TT, 4, 4>, grid, dim3(128), 0, s, xp, wp, yp, m, n, k);
    case 2: return sr::launch_pdl(tc_gemv_bf16_kernel<TT, 8, 2>, grid, dim3(256), 0, s, xp, wp, yp, m, n, k);
    case 3: return sr::launch_pdl(tc_gemv_bf16_kernel<TT, 8, 4>, grid, dim3(256), 0, s, xp, wp, yp, m, n, k);
    default: return cudaErrorInvalidValue;
  }
}

template <int TT>
cudaError_t dispatch_fp8w_tt(const void* x, const void* w, const float* scale, void* y, int m, int n, int k,
                             int variant, cudaStream_t s) {
  const dim3 grid((n + 15) / 16);
  const auto* xp = static_cast<const __nv_bfloat16*>(x);
  const auto* wp = static_cast<const uint8_t*>(w);
  auto* yp = static_cast<__nv_bfloat16*>(y);
  switch (variant) {
    case 0: return sr::launch_pdl(tc_gemv_fp8w_kernel<TT, 4, 2>, grid, dim3(128), 0, s, xp, wp, scale, yp, m, n, k);
    case 1: return sr::launch_pdl(tc_gemv_fp8w_kernel<TT, 4, 4>, grid, dim3(128), 0, s, xp, wp, scale, yp, m, n, k);
    case 2: return sr::launch_pdl(tc_gemv_fp8w_kernel<TT, 8, 2>, grid, dim3(256), 0, s, xp, wp, scale, yp, m, n, k);
    case 3: return sr::launch_pdl(tc_gemv_fp8w_kernel<TT, 8, 4>, grid, dim3(256), 0, s, xp, wp, scale, yp, m, n, k);
    default: return cudaErrorInvalidValue;
  }
}

// Requirements: 1 ≤ m ≤ 64; k a multiple of 32 (BF16) or 64 (FP8); pointers 16-byte aligned, row-major.
inline cudaError_t gemv_bf16(const void* x, const void* w, void* y, int m, int n, int k, int variant,
                             cudaStream_t s) {
  if (k % 32 != 0 || n < 1 || m < 1) return cudaErrorInvalidValue;
  if (m <= 8) return dispatch_bf16_tt<1>(x, w, y, m, n, k, variant, s);
  if (m <= 16) return dispatch_bf16_tt<2>(x, w, y, m, n, k, variant, s);
  if (m <= 32) return dispatch_bf16_tt<4>(x, w, y, m, n, k, variant, s);
  if (m <= kMaxM) return dispatch_bf16_tt<8>(x, w, y, m, n, k, variant, s);
  return cudaErrorInvalidValue;
}

inline cudaError_t gemv_fp8w(const void* x, const void* w, const float* scale, void* y, int m, int n, int k,
                             int variant, cudaStream_t s) {
  if (k % 64 != 0 || n < 1 || m < 1) return cudaErrorInvalidValue;
  if (m <= 8) return dispatch_fp8w_tt<1>(x, w, scale, y, m, n, k, variant, s);
  if (m <= 16) return dispatch_fp8w_tt<2>(x, w, scale, y, m, n, k, variant, s);
  if (m <= 32) return dispatch_fp8w_tt<4>(x, w, scale, y, m, n, k, variant, s);
  if (m <= kMaxM) return dispatch_fp8w_tt<8>(x, w, scale, y, m, n, k, variant, s);
  return cudaErrorInvalidValue;
}

}  // namespace sr::gemv_tc
