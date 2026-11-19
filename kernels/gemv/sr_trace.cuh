// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 F0 Labs

// Per-CTA execution trace: the nanosecond view of a decode step that Nsight cannot give inside a
// CUDA graph.
//
// Built with -DSR_KERNEL_TRACE, every kernel opens a CtaScope as its
// first statement. Thread 0 of each CTA then records one Entry:
// - the launch (%gridid), kernel family, linear block index and the SM it ran on;
// - start and end on the device-wide %globaltimer (ns, comparable across SMs);
// - start and end on the SM's own cycle counter (clock64), for sub-tick resolution.
//
// Cost model, chosen so tracing perturbs what it measures as little as possible:
// - Without SR_KERNEL_TRACE, SR_TRACE_CTA expands to nothing: production kernels are unchanged.
// - With it but tracing off (null buffer), one constant-cache load and a branch in thread 0.
// - With tracing on, thread 0 claims a slot with one atomicAdd on its SM's counter at entry. The result
//   is consumed only at exit, so the atomic's latency overlaps the CTA's work. The start stamps wait
//   in shared memory, so a single register stays live across the kernel.
// - No barriers. The recorded end is warp 0's exit. That equals the CTA's end for kernels whose
//   warp 0 finishes last (reductions); in the CUDA-core GEMV every warp streams
//   the same number of rows, so the skew is a few memory latencies at most.
//
// Each translation unit has its own copy of the trace state (no relocatable device code), so each
// exports a configure function (SR_TRACE_EXPORT_UNIT) and the host program calls each one. The
// PyTorch binding in this repository does not enable tracing.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace sr::trace {

inline constexpr int kMaxSms = 256;  // %smid can exceed the SM count on floorswept parts

// Fixed IDs, decoded by trace readers. Not every ID has a kernel in this repository.
enum KernelId : uint16_t {
  kGemvCudaCoreBf16 = 1,
  kGemvCudaCoreFp8 = 2,
  kGemvTensorCoreBf16 = 3,
  kGemvTensorCoreFp8 = 4,
  kAttentionDecode = 5,
  kAttentionCombine = 6,
  kAttentionPrefill = 7,
  kEmbedding = 8,
  kRmsnorm = 9,
  kAddRmsnorm = 10,
  kSiluMul = 11,
  kArgmaxPartial = 12,
  kArgmaxFinal = 13,
  kQkNormRopeAppend = 14,
  kGemmTensorCoreWideFp8 = 15,
};

struct alignas(16) Entry {
  uint64_t grid;
  uint64_t start_ns;
  uint64_t end_ns;
  uint64_t start_clk;
  uint64_t end_clk;
  uint32_t block;
  uint16_t kernel;
  uint16_t sm;
};
static_assert(sizeof(Entry) == 48, "the host reads 48-byte entries");

struct State {
  Entry* entries;      // [kMaxSms][per_sm]; null: tracing off
  uint32_t* counters;  // [kMaxSms]; entries claimed per SM, may exceed per_sm (overflow is dropped)
  uint32_t per_sm;
};

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ uint32_t smid() {
  uint32_t s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}

__device__ __forceinline__ uint64_t gridid() {
  uint64_t g;
  asm volatile("mov.u64 %0, %%gridid;" : "=l"(g));
  return g;
}

}  // namespace sr::trace

#ifdef SR_KERNEL_TRACE

namespace sr::trace {

static __constant__ State g_state;

struct StartStamp {
  uint64_t ns;
  uint64_t clk;
  uint32_t sm;
};

__device__ __forceinline__ StartStamp& start_stamp() {
  __shared__ StartStamp stamp;
  return stamp;
}

class CtaScope {
 public:
  __device__ __forceinline__ explicit CtaScope(uint16_t kernel) : kernel_(kernel) {
    if ((threadIdx.x | threadIdx.y | threadIdx.z) != 0 || g_state.entries == nullptr) return;
    StartStamp& s = start_stamp();
    s.clk = clock64();
    s.ns = globaltimer();
    s.sm = smid() & (kMaxSms - 1);
    slot_ = atomicAdd(g_state.counters + s.sm, 1u);
  }

  __device__ __forceinline__ ~CtaScope() {
    if (slot_ == kOff) return;
    const uint64_t end_clk = clock64();
    const uint64_t end_ns = globaltimer();
    const StartStamp& s = start_stamp();
    if (slot_ >= g_state.per_sm) return;
    Entry e;
    e.grid = gridid();
    e.start_ns = s.ns;
    e.end_ns = end_ns;
    e.start_clk = s.clk;
    e.end_clk = end_clk;
    e.block = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    e.kernel = kernel_;
    e.sm = static_cast<uint16_t>(s.sm);
    g_state.entries[static_cast<size_t>(s.sm) * g_state.per_sm + slot_] = e;
  }

  CtaScope(const CtaScope&) = delete;
  CtaScope& operator=(const CtaScope&) = delete;

 private:
  static constexpr uint32_t kOff = 0xffffffffu;
  uint32_t slot_ = kOff;
  uint16_t kernel_;
};

// Points this translation unit's kernels at a trace buffer (null entries: off).
static inline cudaError_t configure_unit(void* entries, void* counters, uint32_t per_sm) {
  const State s{static_cast<Entry*>(entries), static_cast<uint32_t*>(counters), per_sm};
  return cudaMemcpyToSymbol(g_state, &s, sizeof(s));
}

}  // namespace sr::trace

#define SR_TRACE_CTA(kernel_id) const ::sr::trace::CtaScope sr_trace_cta_scope_(kernel_id)
#define SR_TRACE_EXPORT_UNIT(unit)                                                              \
  extern "C" int sr_trace_configure_##unit(void* entries, void* counters, uint32_t per_sm) { \
    return static_cast<int>(::sr::trace::configure_unit(entries, counters, per_sm));          \
  }

#else

#define SR_TRACE_CTA(kernel_id) \
  do {                          \
  } while (0)
#define SR_TRACE_EXPORT_UNIT(unit)                                                     \
  extern "C" int sr_trace_configure_##unit(void*, void*, uint32_t) {                 \
    return static_cast<int>(cudaErrorNotSupported);                                  \
  }

#endif
