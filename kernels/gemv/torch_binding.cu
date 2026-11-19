// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 F0 Labs

// PyTorch binding of the decode GEMVs, built as `sr_vllm._C` by setup.py. Callers: the vLLM plugin
// (src/sr_vllm) and the kernel benchmark (benchmarks/kernel_sweep.py). Code without PyTorch can include
// the headers and call their launch functions directly.
// c10's stream header avoids ATen/cuda/CUDAContext.h, which drags in cuSPARSE/cuBLAS headers that
// minimal CUDA toolkits (e.g. the vLLM container) don't ship.
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "sr_gemv.cuh"
#include "sr_gemm_tc_wide.cuh"
#include "sr_gemv_tc.cuh"

// The SM architectures this binary was compiled for, e.g. "90a,120" (setup.py). Empty when the build
// did not say.
#ifndef SR_CUDA_ARCHS
#define SR_CUDA_ARCHS ""
#endif

namespace {

void check(cudaError_t err) { TORCH_CHECK(err == cudaSuccess, "sr_gemv launch failed: ", cudaGetErrorString(err)); }

torch::Tensor gemv_bf16(torch::Tensor x, torch::Tensor w, int64_t variant) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x and w must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16, "x and w must be bf16");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "x and w must be contiguous");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "shapes must be x[M,K], w[N,K]");
  auto y = torch::empty({x.size(0), w.size(0)}, x.options());
  check(sr::gemv::gemv_bf16(x.data_ptr(), w.data_ptr(), y.data_ptr(), static_cast<int>(x.size(0)),
                            static_cast<int>(w.size(0)), static_cast<int>(x.size(1)), static_cast<int>(variant),
                            c10::cuda::getCurrentCUDAStream().stream()));
  return y;
}

torch::Tensor gemv_fp8w(torch::Tensor x, torch::Tensor w, torch::Tensor scale, int64_t variant) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && scale.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
  TORCH_CHECK(w.scalar_type() == at::ScalarType::Float8_e4m3fn, "w must be float8_e4m3fn");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32 && scale.numel() == w.size(0), "scale must be fp32 [N]");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && scale.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "shapes must be x[M,K], w[N,K]");
  auto y = torch::empty({x.size(0), w.size(0)}, x.options());
  check(sr::gemv::gemv_fp8w(x.data_ptr(), w.data_ptr(), scale.data_ptr<float>(), y.data_ptr(),
                            static_cast<int>(x.size(0)), static_cast<int>(w.size(0)), static_cast<int>(x.size(1)),
                            static_cast<int>(variant), c10::cuda::getCurrentCUDAStream().stream()));
  return y;
}

torch::Tensor tc_gemv_bf16(torch::Tensor x, torch::Tensor w, int64_t variant) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x and w must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16, "x and w must be bf16");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "x and w must be contiguous");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "shapes must be x[M,K], w[N,K]");
  auto y = torch::empty({x.size(0), w.size(0)}, x.options());
  check(sr::gemv_tc::gemv_bf16(x.data_ptr(), w.data_ptr(), y.data_ptr(), static_cast<int>(x.size(0)),
                               static_cast<int>(w.size(0)), static_cast<int>(x.size(1)), static_cast<int>(variant),
                               c10::cuda::getCurrentCUDAStream().stream()));
  return y;
}

torch::Tensor tc_gemv_fp8w(torch::Tensor x, torch::Tensor w, torch::Tensor scale, int64_t variant) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && scale.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
  TORCH_CHECK(w.scalar_type() == at::ScalarType::Float8_e4m3fn, "w must be float8_e4m3fn");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32 && scale.numel() == w.size(0), "scale must be fp32 [N]");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && scale.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "shapes must be x[M,K], w[N,K]");
  auto y = torch::empty({x.size(0), w.size(0)}, x.options());
  check(sr::gemv_tc::gemv_fp8w(x.data_ptr(), w.data_ptr(), scale.data_ptr<float>(), y.data_ptr(),
                               static_cast<int>(x.size(0)), static_cast<int>(w.size(0)), static_cast<int>(x.size(1)),
                               static_cast<int>(variant), c10::cuda::getCurrentCUDAStream().stream()));
  return y;
}

torch::Tensor tc_wide_fp8w(torch::Tensor x, torch::Tensor w, torch::Tensor scale, torch::Tensor ws,
                           torch::Tensor counters, int64_t split, int64_t variant) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && scale.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
  TORCH_CHECK(w.scalar_type() == at::ScalarType::Float8_e4m3fn, "w must be float8_e4m3fn");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat32 && scale.numel() == w.size(0), "scale must be fp32 [N]");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && scale.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1), "shapes must be x[M,K], w[N,K]");
  const int m = static_cast<int>(x.size(0));
  const int n = static_cast<int>(w.size(0));
  float* ws_ptr = nullptr;
  int* counter_ptr = nullptr;
  if (split > 1) {
    TORCH_CHECK(ws.is_cuda() && ws.scalar_type() == torch::kFloat32 && ws.is_contiguous(), "ws must be fp32 CUDA");
    TORCH_CHECK(static_cast<size_t>(ws.numel()) >= sr::gemm_tc_wide::workspace_floats(m, n, static_cast<int>(split)),
                "ws too small for this split");
    TORCH_CHECK(counters.is_cuda() && counters.scalar_type() == torch::kInt32 && counters.is_contiguous(),
                "counters must be int32 CUDA");
    TORCH_CHECK(counters.numel() >= sr::gemm_tc_wide::counter_count(n), "counters too small");
    ws_ptr = ws.data_ptr<float>();
    counter_ptr = counters.data_ptr<int>();
  }
  auto y = torch::empty({m, n}, x.options());
  check(sr::gemm_tc_wide::gemm_fp8w(x.data_ptr(), w.data_ptr(), scale.data_ptr<float>(), y.data_ptr(), ws_ptr,
                                    counter_ptr, m, n, static_cast<int>(x.size(1)), static_cast<int>(split),
                                    static_cast<int>(variant), c10::cuda::getCurrentCUDAStream().stream()));
  return y;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv_bf16", &gemv_bf16, "y = x @ w.T, bf16 weights, small M");
  m.def("gemv_fp8w", &gemv_fp8w, "y = (x @ w.T) * scale, fp8 e4m3 weights with per-row scale, bf16 activations");
  m.def("tc_gemv_bf16", &tc_gemv_bf16, "tensor-core skinny GEMM, bf16 weights, M <= 64");
  m.def("tc_gemv_fp8w", &tc_gemv_fp8w, "tensor-core skinny GEMM, fp8 e4m3 weights + per-row scale, bf16 activations, M <= 64");
  m.def("tc_wide_fp8w", &tc_wide_fp8w,
        "wide tensor-core GEMM, fp8 e4m3 weights + per-row scale, bf16 activations, M <= 64, deterministic split-K");
  m.attr("cuda_archs") = SR_CUDA_ARCHS;
  m.attr("num_variants") = sr::gemv::kNumVariants;
  m.attr("tc_num_variants") = sr::gemv_tc::kNumVariants;
  m.attr("max_m") = sr::gemv::kMaxM;
  m.attr("tc_max_m") = sr::gemv_tc::kMaxM;
  m.attr("wide_num_variants") = sr::gemm_tc_wide::kNumVariants;
  m.attr("wide_max_m") = sr::gemm_tc_wide::kMaxM;
  m.attr("wide_max_split") = sr::gemm_tc_wide::kMaxSplit;
  m.attr("wide_min_channels_per_cta") = sr::gemm_tc_wide::kMinChannelsPerCta;
}
