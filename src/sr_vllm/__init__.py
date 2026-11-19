# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 F0 Labs

"""rabbit-gemm's decode kernels inside stock vLLM, as the `sr_w8a16` quantization method.

vLLM calls `register` in every process it starts, through the `vllm.general_plugins` entry point.
"""

QUANTIZATION_NAME = "sr_w8a16"


def register() -> None:
    """Registers the `sr_w8a16` quantization method with vLLM. Safe to call more than once."""
    # Deferred so `import sr_vllm` stays cheap; the extension and vLLM internals load only when vLLM asks.
    from sr_vllm.w8a16 import register_quantization  # noqa: PLC0415

    register_quantization()
