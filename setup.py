# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 F0 Labs

"""Builds `sr_vllm._C`, the PyTorch binding of kernels/gemv, against the torch of the serving environment.

Install it into the environment vLLM runs in with `pip install --no-build-isolation .`, or build a wheel for
that environment with `pip wheel --no-build-isolation --no-deps .`: the extension must compile against that
exact torch, so build isolation stays off.
"""

import os
import sys

from setuptools import setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    raise SystemExit(
        "rabbit-gemm compiles against the torch it will run with: install vLLM (and with it torch) "
        "first, then build with `pip install --no-build-isolation .`"
    ) from None

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from sr_vllm.archs import DEFAULT_ARCHS, parse_archs  # the package's own, light module

ROOT = os.path.dirname(os.path.abspath(__file__))

# SM architectures the binary carries, comma-separated.
ARCH_TEXT = os.environ.get("SR_CUDA_ARCH", DEFAULT_ARCHS)
try:
    ARCHS = parse_archs(ARCH_TEXT)
except ValueError as err:
    raise SystemExit(f"SR_CUDA_ARCH: {err}") from None

KERNELS = os.path.join("kernels", "gemv")

setup(
    ext_modules=[
        CUDAExtension(
            name="sr_vllm._C",
            sources=[os.path.join(KERNELS, "torch_binding.cu")],
            include_dirs=[os.path.join(ROOT, KERNELS)],
            # The binary says what it was built for; the plugin checks the GPU against it at load.
            define_macros=[("SR_CUDA_ARCHS", f'"{",".join(a.name for a in ARCHS)}"')],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                # torch's builder adds no architecture of its own once it sees a -gencode flag here, so the
                # GPU that happens to be visible at build time never decides what the binary runs on.
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-lineinfo",
                    "--expt-relaxed-constexpr",
                    *(a.gencode for a in ARCHS),
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
