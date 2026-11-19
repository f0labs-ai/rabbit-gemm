// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 F0 Labs

// Wide tensor-core skinny GEMM for decode batches: y[M, N] = (x[M, K] · W[N, K]^T) * scale[N], 1 ≤ M ≤ 64,
// FP8 (E4M3) weights with one f32 scale per output row, bf16 activations (W8A16).
//
// Why a second tensor-core kernel. sr_gemv_tc.cuh gives each CTA 16 output channels, so every CTA re-reads
// and re-converts all of x: M / 8 activation bytes per weight byte. At M = 32 that is 4 bytes, about 96 of
// the ~150 instructions spent per KiB of weights are bf16 → f16 conversions, the kernel needs 126 registers
// and it streams at 49% of HBM bandwidth on an H100. This kernel:
//
//  * gives a CTA WN warps × 16 = 64 or 128 output channels that all walk the same K range, so x is
//    converted once per CTA into shared memory: M / (16 · WN) activation values per weight byte;
//  * stages x in chunks of CB k-blocks (64 k each), double-buffered: chunk c + 1 is loaded into registers
//    and the weights of chunk c + 1 are in flight while the tensor cores work on chunk c;
//  * splits K across `split` CTAs (gridDim.y) when N alone cannot fill the SMs (the 4,096-wide o and down
//    projections are 64 CTAs of 64 channels). Each split writes an f32 partial tile; the last CTA to finish
//    a tile (an atomic counter) sums the partials in split order and writes y, so the result is bit-identical
//    from run to run, then resets the counter for the next call.
//
// The MMA mapping, K permutation and E4M3 → f16 expansion are those of sr_gemv_tc.cuh ("swap AB": weights are
// the 16-row A operand, up to 8 tokens the B operand). A lane's 16 bytes of a 64-byte k-block feed four
// k-steps; the activations it pairs with are the 16 consecutive k at the same offset, read from shared
// memory as two 16-byte loads. Shared rows carry 16 bytes of padding, which makes those loads conflict-free.
//
// Budget: one call per projection per decode step; target ≥ 85% of HBM bandwidth at M = 16–64
// (gate/up at M = 32: 100.7 MB, floor 31.6 µs at 3,185 GB/s).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "sr_gemv_tc.cuh"
#include "sr_launch.cuh"

namespace sr::gemm_tc_wide {

inline constexpr int kMaxM = 64;
inline constexpr int kMaxSplit = 16;
// (WN, CB): warps (16 channels each) per CTA and 64-k blocks per pipeline chunk.
inline constexpr int kNumVariants = 4;  // {(4,1), (4,2), (8,1), (8,2)}
inline constexpr int kChannelsPerWarp = 16;

// Smallest CTA tile, in output channels, over all variants: sizes the split-K counter array.
inline constexpr int kMinChannelsPerCta = 4 * kChannelsPerWarp;

template <int TT, int WN, int CB>
__global__ void __launch_bounds__(WN * 32)
    tc_wide_fp8w_kernel(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w,
                        const float* __restrict__ scale, __nv_bfloat16* __restrict__ y, float* __restrict__ ws,
                        int* __restrict__ counters, int m, int n, int k, int split) {
  SR_TRACE_CTA(sr::trace::kGemmTensorCoreWideFp8);
  sr::pdl_trigger();
  constexpr int kThreads = WN * 32;
  constexpr int kTokens = 8 * TT;
  constexpr int kChunkK = 64 * CB;
  constexpr int kRowHalfs = kChunkK + 8;               // +16 bytes per row: conflict-free 16-byte reads
  constexpr int kRowVecs = kChunkK / 8;                // 16-byte vectors per token row of a chunk
  constexpr int kChunkVecs = kTokens * kRowVecs;
  constexpr int kVecsPerThread = (kChunkVecs + kThreads - 1) / kThreads;
  __shared__ __align__(16) __half xs[2][kTokens][kRowHalfs];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int n0 = blockIdx.x * (WN * kChannelsPerWarp) + warp * kChannelsPerWarp;

  const int chunks = k / kChunkK;
  const int per_split = (chunks + split - 1) / split;
  const int c_begin = blockIdx.y * per_split;
  const int c_end = min(chunks, c_begin + per_split);

  // Row pointers for this lane's two channels, clamped so tail warps read valid memory (never stored).
  const uint4* wa = reinterpret_cast<const uint4*>(w + static_cast<size_t>(min(n0 + g, n - 1)) * k) + t;
  const uint4* wb = reinterpret_cast<const uint4*>(w + static_cast<size_t>(min(n0 + g + 8, n - 1)) * k) + t;
  const uint4* x4 = reinterpret_cast<const uint4*>(x);
  const int x_row_vecs = k / 8;

  uint4 wcur[CB][2], wnext[CB][2];
  uint4 xr[kVecsPerThread];

  auto load_weights = [&](uint4 (&dst)[CB][2], int chunk) {
#pragma unroll
    for (int b = 0; b < CB; ++b) {
      const size_t kb = static_cast<size_t>(chunk) * CB + b;
      dst[b][0] = __ldcs(wa + kb * 4);
      dst[b][1] = __ldcs(wb + kb * 4);
    }
  };
  auto load_x = [&](int chunk) {
#pragma unroll
    for (int j = 0; j < kVecsPerThread; ++j) {
      const int i = threadIdx.x + j * kThreads;
      const int tok = i / kRowVecs;
      const int col = i % kRowVecs;
      xr[j] = (i < kChunkVecs && tok < m)
                  ? __ldg(x4 + static_cast<size_t>(tok) * x_row_vecs + static_cast<size_t>(chunk) * kRowVecs + col)
                  : make_uint4(0, 0, 0, 0);
    }
  };
  auto store_x = [&](int buf) {
#pragma unroll
    for (int j = 0; j < kVecsPerThread; ++j) {
      const int i = threadIdx.x + j * kThreads;
      if (i < kChunkVecs) {
        const uint4 v = xr[j];
        const uint4 h = make_uint4(sr::gemv_tc::bf16x2_to_f16x2(v.x), sr::gemv_tc::bf16x2_to_f16x2(v.y),
                                   sr::gemv_tc::bf16x2_to_f16x2(v.z), sr::gemv_tc::bf16x2_to_f16x2(v.w));
        *reinterpret_cast<uint4*>(&xs[buf][i / kRowVecs][(i % kRowVecs) * 8]) = h;
      }
    }
  };

  float c[TT][4];
#pragma unroll
  for (int tt = 0; tt < TT; ++tt)
#pragma unroll
    for (int i = 0; i < 4; ++i) c[tt][i] = 0.f;

  if (c_begin < c_end) load_weights(wcur, c_begin);  // weights never depend on the previous kernel
  sr::pdl_wait();
  if (c_begin < c_end) {
    load_x(c_begin);
    store_x(0);
  }
  __syncthreads();

  for (int chunk = c_begin; chunk < c_end; ++chunk) {
    const int buf = (chunk - c_begin) & 1;
    const bool more = chunk + 1 < c_end;
    if (more) {  // next chunk's weights and activations are in flight during this chunk's math
      load_weights(wnext, chunk + 1);
      load_x(chunk + 1);
    }
#pragma unroll
    for (int b = 0; b < CB; ++b) {
      const uint32_t wlo[4] = {wcur[b][0].x, wcur[b][0].y, wcur[b][0].z, wcur[b][0].w};
      const uint32_t whi[4] = {wcur[b][1].x, wcur[b][1].y, wcur[b][1].z, wcur[b][1].w};
      uint32_t a[4][4];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        a[s][0] = sr::gemv_tc::e4m3x2_to_f16x2(wlo[s]);
        a[s][1] = sr::gemv_tc::e4m3x2_to_f16x2(whi[s]);
        a[s][2] = sr::gemv_tc::e4m3x2_to_f16x2(wlo[s] >> 16);
        a[s][3] = sr::gemv_tc::e4m3x2_to_f16x2(whi[s] >> 16);
      }
#pragma unroll
      for (int tt = 0; tt < TT; ++tt) {
        const uint4* xp = reinterpret_cast<const uint4*>(&xs[buf][tt * 8 + g][b * 64 + 16 * t]);
        const uint4 xa = xp[0];
        const uint4 xb = xp[1];
        const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const uint32_t bfrag[2] = {xw[2 * s], xw[2 * s + 1]};
          sr::gemv_tc::mma_f16(c[tt], a[s], bfrag);
        }
      }
    }
    if (more) {
      store_x(buf ^ 1);  // buf ^ 1 was last read before the previous barrier
#pragma unroll
      for (int b = 0; b < CB; ++b) {
        wcur[b][0] = wnext[b][0];
        wcur[b][1] = wnext[b][1];
      }
    }
    __syncthreads();
  }

  // C fragment: c0, c1 → (channel g, tokens 2t, 2t+1); c2, c3 → (channel g + 8, tokens 2t, 2t+1).
  const int row_lo = n0 + g;
  const int row_hi = n0 + g + 8;
  if (split == 1) {
    const float s_lo = row_lo < n ? scale[row_lo] : 0.f;
    const float s_hi = row_hi < n ? scale[row_hi] : 0.f;
#pragma unroll
    for (int tt = 0; tt < TT; ++tt) {
      const int tok = tt * 8 + 2 * t;
      if (row_lo < n) {
        if (tok < m) y[static_cast<size_t>(tok) * n + row_lo] = __float2bfloat16(c[tt][0] * s_lo);
        if (tok + 1 < m) y[static_cast<size_t>(tok + 1) * n + row_lo] = __float2bfloat16(c[tt][1] * s_lo);
      }
      if (row_hi < n) {
        if (tok < m) y[static_cast<size_t>(tok) * n + row_hi] = __float2bfloat16(c[tt][2] * s_hi);
        if (tok + 1 < m) y[static_cast<size_t>(tok + 1) * n + row_hi] = __float2bfloat16(c[tt][3] * s_hi);
      }
    }
    return;
  }

  // Split-K: write this split's partial tile, then let the last CTA of the tile reduce in split order.
  float* part = ws + static_cast<size_t>(blockIdx.y) * m * n;
#pragma unroll
  for (int tt = 0; tt < TT; ++tt) {
    const int tok = tt * 8 + 2 * t;
    if (row_lo < n) {
      if (tok < m) part[static_cast<size_t>(tok) * n + row_lo] = c[tt][0];
      if (tok + 1 < m) part[static_cast<size_t>(tok + 1) * n + row_lo] = c[tt][1];
    }
    if (row_hi < n) {
      if (tok < m) part[static_cast<size_t>(tok) * n + row_hi] = c[tt][2];
      if (tok + 1 < m) part[static_cast<size_t>(tok + 1) * n + row_hi] = c[tt][3];
    }
  }
  __threadfence();  // partials visible device-wide before the counter says so
  __syncthreads();
  __shared__ int is_last;
  if (threadIdx.x == 0) is_last = atomicAdd(&counters[blockIdx.x], 1) == split - 1;
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  const float s_lo = row_lo < n ? scale[row_lo] : 0.f;
  const float s_hi = row_hi < n ? scale[row_hi] : 0.f;
  const size_t split_stride = static_cast<size_t>(m) * n;
#pragma unroll
  for (int tt = 0; tt < TT; ++tt) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int tok = tt * 8 + 2 * t + (i & 1);
      const int row = (i < 2) ? row_lo : row_hi;
      if (tok >= m || row >= n) continue;
      const size_t at = static_cast<size_t>(tok) * n + row;
      float acc = 0.f;
      for (int sp = 0; sp < split; ++sp) acc += __ldcg(ws + sp * split_stride + at);
      y[at] = __float2bfloat16(acc * (i < 2 ? s_lo : s_hi));
    }
  }
  if (threadIdx.x == 0) counters[blockIdx.x] = 0;  // ready for the next call in stream order
}

template <int TT>
cudaError_t dispatch_tt(const void* x, const void* w, const float* scale, void* y, float* ws, int* counters,
                        int m, int n, int k, int split, int variant, cudaStream_t s) {
  const auto* xp = static_cast<const __nv_bfloat16*>(x);
  const auto* wp = static_cast<const uint8_t*>(w);
  auto* yp = static_cast<__nv_bfloat16*>(y);
  switch (variant) {
    case 0: return sr::launch_pdl(tc_wide_fp8w_kernel<TT, 4, 1>, dim3((n + 63) / 64, split), dim3(128), 0, s, xp,
                                  wp, scale, yp, ws, counters, m, n, k, split);
    case 1: return sr::launch_pdl(tc_wide_fp8w_kernel<TT, 4, 2>, dim3((n + 63) / 64, split), dim3(128), 0, s, xp,
                                  wp, scale, yp, ws, counters, m, n, k, split);
    case 2: return sr::launch_pdl(tc_wide_fp8w_kernel<TT, 8, 1>, dim3((n + 127) / 128, split), dim3(256), 0, s, xp,
                                  wp, scale, yp, ws, counters, m, n, k, split);
    case 3: return sr::launch_pdl(tc_wide_fp8w_kernel<TT, 8, 2>, dim3((n + 127) / 128, split), dim3(256), 0, s, xp,
                                  wp, scale, yp, ws, counters, m, n, k, split);
    default: return cudaErrorInvalidValue;
  }
}

// Floats of split-K workspace a call needs; zero when split == 1.
inline size_t workspace_floats(int m, int n, int split) {
  return split > 1 ? static_cast<size_t>(split) * m * n : 0;
}

// Split-K tile counters a call needs (they must start at zero; every call leaves them at zero).
inline int counter_count(int n) { return (n + kMinChannelsPerCta - 1) / kMinChannelsPerCta; }

// Requirements: 1 ≤ m ≤ 64; k a multiple of 128; 1 ≤ split ≤ min(kMaxSplit, k / (64 · CB)); for split > 1,
// ws holds workspace_floats(m, n, split) floats and counters holds counter_count(n) zeros. Row-major x, w, y;
// 16-byte aligned pointers.
inline cudaError_t gemm_fp8w(const void* x, const void* w, const float* scale, void* y, float* ws, int* counters,
                             int m, int n, int k, int split, int variant, cudaStream_t s) {
  const int cb = (variant == 1 || variant == 3) ? 2 : 1;
  if (m < 1 || m > kMaxM || n < 1 || k % 128 != 0 || split < 1 || split > kMaxSplit || split > k / (64 * cb))
    return cudaErrorInvalidValue;
  if (split > 1 && (ws == nullptr || counters == nullptr)) return cudaErrorInvalidValue;
  if (m <= 8) return dispatch_tt<1>(x, w, scale, y, ws, counters, m, n, k, split, variant, s);
  if (m <= 16) return dispatch_tt<2>(x, w, scale, y, ws, counters, m, n, k, split, variant, s);
  if (m <= 24) return dispatch_tt<3>(x, w, scale, y, ws, counters, m, n, k, split, variant, s);
  if (m <= 32) return dispatch_tt<4>(x, w, scale, y, ws, counters, m, n, k, split, variant, s);
  if (m <= 48) return dispatch_tt<6>(x, w, scale, y, ws, counters, m, n, k, split, variant, s);
  return dispatch_tt<8>(x, w, scale, y, ws, counters, m, n, k, split, variant, s);
}

}  // namespace sr::gemm_tc_wide
