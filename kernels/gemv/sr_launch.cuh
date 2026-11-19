// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 F0 Labs

// Programmatic dependent launch (PDL, sm_90+).
//
// Every kernel here calls pdl_trigger() first, which lets the next kernel in the stream start
// launching immediately. It calls pdl_wait() before it reads anything an earlier kernel wrote.
// Work that doesn't depend on the predecessor goes before the wait and overlaps its tail: in the
// GEMVs, that is the weight streaming.
//
// Safety:
// - griddepcontrol.wait returns only once every prerequisite grid has completed and its memory is
//   visible, so triggering early is always correct.
// - A dependent grid launches only after every CTA of the primary has triggered or exited, so
//   dependents can never take SM slots that unscheduled primary CTAs still need.
// - Without the launch attribute, or across non-PDL work such as cuBLAS kernels or memcpy nodes,
//   both instructions are no-ops and ordering is ordinary stream order.
#pragma once

#include <cuda_runtime.h>

#include <utility>

#include "sr_trace.cuh"

namespace sr {

__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }

// Launches `kernel` with programmatic stream serialization allowed (captured as a programmatic edge in
// CUDA graphs). Returns the launch status.
template <typename... Params, typename... Args>
cudaError_t launch_pdl(void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
                       Args&&... args) {
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attr;
#ifdef SR_ENABLE_PDL
  config.numAttrs = 1;
#else
  // Off by default: on a power-capped (500 W) H100 it gave no gain within the ±3% run-to-run noise.
  // Build with -DSR_ENABLE_PDL to measure it again, e.g. on an uncapped part.
  config.numAttrs = 0;
#endif
  return cudaLaunchKernelEx(&config, kernel, std::forward<Args>(args)...);
}

}  // namespace sr
