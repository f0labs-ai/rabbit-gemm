#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 F0 Labs
#
# Builds, installs and checks rabbit-gemm inside vllm/vllm-openai:v0.32.0, the environment it targets:
#
#   docker run --rm -v "$PWD:/src" -w /src --entrypoint bash vllm/vllm-openai:v0.32.0 ci/check_in_vllm_image.sh
#
# Without a GPU (as in CI) the GPU tests are skipped; on a GPU host the same script runs them too.
set -euo pipefail

# The image's Python is the serving environment, so the extension is built against its torch.
export PIP_BREAK_SYSTEM_PACKAGES=1
python3 -m pip install --no-build-isolation --no-deps .
python3 -m pip install ruff==0.17.6 mypy==2.4.1 pytest==9.2.0

python3 -m ruff check .
python3 -m ruff format --check .
python3 -m mypy
python3 -m pytest -q -rs
