# rabbit-gemm

Weight-only FP8 matrix-multiply kernels that make LLM decoding faster at low concurrency on NVIDIA H100 and RTX PRO 6000 Blackwell GPUs, and a plugin that adds them to unmodified vLLM.

## Why

A language model writes its output one token at a time. When a GPU serves only a few requests at once, each step's speed is set by how fast the GPU can read the model's weights from memory. Storing weights in 8-bit floating point (FP8) instead of 16 bits halves that traffic.

vLLM's standard FP8 path also converts each layer's input to FP8 before every matrix multiply, so that it can use the GPU's FP8 tensor cores. With few requests in flight those units sit mostly idle, and the conversion is pure overhead: an extra pass over the data before every layer, plus rounding error.

rabbit-gemm's kernels read FP8 weights and keep the inputs in 16-bit BF16 (W8A16), so nothing is converted at run time. The `sr_w8a16` plugin adds them to vLLM and, for every layer and batch size, uses whichever is faster: our kernels or vLLM's standard FP8 kernel.

## Results

Time per generated token with one request, vLLM 0.32.0:

| Model | GPU | Standard FP8 | With rabbit-gemm | Change |
|---|---|---:|---:|---:|
| Qwen3-8B | H100 | 4.51 ms | 3.97 ms | −12.0% |
| Qwen3-8B | RTX PRO 6000 | 6.48 ms | 6.00 ms | −7.4% |
| Qwen3.8-27B | H100 | 12.56 ms | 11.50 ms | −8.4% |
| Qwen3.8-27B | RTX PRO 6000 | 19.75 ms | 18.35 ms | −7.1% |

Change in time per token as more requests share the GPU (negative is faster; values in italics are within run-to-run noise):

| Model | GPU | 1 | 4 | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | H100 | −12.0% | −3.2% | −5.2% | *−0.4%* | +2.8% | +2.7% | +2.6% | +2.5% |
| Qwen3-8B | RTX PRO 6000 | −7.4% | −5.4% | −4.5% | −2.3% | +2.0% | +2.0% | +2.0% | +3.8% |
| Qwen3.8-27B | H100 | −8.4% | −0.5% | *+0.1%* | *−0.1%* | +10.6% | +2.7% | +2.5% | +1.8% |
| Qwen3.8-27B | RTX PRO 6000 | −7.1% | −11.1% | −5.2% | −1.6% | +1.0% | +1.1% | *+1.6%* | *+1.8%* |

- **Baseline.** "Standard FP8" is vLLM's default `--quantization fp8` for Qwen3-8B, and Qwen's published FP8 checkpoint for Qwen3.8-27B. On the H100 that checkpoint runs on DeepGEMM, which is unusually fast at 24 requests; that explains the +10.6%.
- **Where the gain comes from.** Mostly not from faster multiplication but from removing the conversion work around it. At one token, a call to our kernel has about 3 µs of fixed cost before it streams the weights, against 8 µs for vLLM's FP8 kernel including its input conversion (H100; 2 µs against 6 µs on the RTX PRO 6000). Both stream weights at about the same rate.
- **Output quality.** With greedy decoding, the generated text matches the 16-bit model exactly in 33 of 256 sequences, against 14 for vLLM's FP8 kernel on the same weights. An average measure of the difference (KL divergence) cannot tell the FP8 variants apart.

The method, the measurements and every caveat are in the paper (see [Citation](#citation)).

## Requirements

- Linux, and an NVIDIA GPU of architecture sm_90 (H100, H200) or sm_120 (RTX 5090, RTX PRO 6000 Blackwell).
- vLLM 0.32.0 with the torch it ships with (2.13).
- A CUDA toolkit with `nvcc` whose major version matches torch's CUDA (13.x for vLLM 0.32.0), and setuptools 77 or newer.

The `vllm/vllm-openai:v0.32.0` container image has all of these.

## Install

```bash
git clone https://github.com/f0labs-ai/rabbit-gemm
cd rabbit-gemm
pip install --no-build-isolation .
```

The kernels compile against the torch already installed, so build isolation must stay off. The build takes a few minutes. By default it produces code for sm_90a and sm_120. To build only what you need, set `SR_CUDA_ARCH`, for example `SR_CUDA_ARCH=90a pip install --no-build-isolation .`.

## Use

```bash
vllm serve Qwen/Qwen3-8B --quantization sr_w8a16
```

vLLM finds the plugin through its plugin entry point, so no code changes are needed. Point it at a 16-bit (BF16) checkpoint: the plugin converts the weights to FP8 as the model loads.

At startup the plugin times the candidate kernels for every distinct layer shape and batch size, and logs one line per shape with its choices, starting `sr_w8a16 plan N=… K=…`.

## How it works

- **Weights.** Every linear layer is held as FP8 (E4M3) with one scale per output row, using vLLM's own conversion code, so the weight bytes are the same ones vLLM's FP8 kernels read.
- **Kernels.** There are two of ours:
  - a CUDA-core kernel for up to 8 tokens;
  - a tensor-core kernel (`mma.sync`) for up to 64 tokens.

  Both stream weights from memory once per step and expand FP8 to 16 bits in registers.
- **Choice.** At startup, each layer shape is timed at each batch size vLLM captures as a CUDA graph (up to 64 tokens). For each one, the plugin keeps the fastest of our two kernels and vLLM's CUTLASS FP8 kernel, the latter timed including its input conversion. Inputs larger than the biggest timed batch, which includes most prompts, always use vLLM's kernel.
- **Compilation.** The plugin's operator is opaque to `torch.compile`, so the choice is made on the real batch size at run time.

| Variable | Values | Effect |
|---|---|---|
| `SR_VLLM_GEMV_POLICY` | `auto` (default) | The fastest kernel per layer shape and batch size. |
| | `sr` | Our kernels wherever they support the shape. |
| | `cutlass` | vLLM's FP8 kernel everywhere, through the plugin. Use it to measure the plugin's own overhead. |
| `SR_CUDA_ARCH` | `90a,120` (default) | GPU architectures to compile for. Read at build time. |

## Limitations

- **It only helps at low concurrency.** The gain is gone by about 16 concurrent requests, and beyond that the plugin is 1 to 4% slower than standard vLLM. There the plugin hands the work to vLLM's FP8 kernel, but the input conversion then runs as separate kernels behind the plugin's operator, where standard vLLM merges it into the preceding normalization. If you serve at high concurrency, use `--quantization fp8`.
- **It loads 16-bit checkpoints only.** It converts weights itself at load, so it cannot read pre-quantized FP8 checkpoints. Inputs must be BF16. Embeddings and the output head stay in 16 bits, as with vLLM's online FP8.
- **It is tied to vLLM 0.32.0 and torch 2.13.** The plugin builds on an internal vLLM class, which can change between releases.
- **Testing is limited.** It was measured on H100 and RTX PRO 6000 Blackwell GPUs, with Qwen3-8B and Qwen3.8-27B. The kernels need sm_90 or newer. Other architectures, such as sm_100, are untested.
- **Startup takes longer,** because of the timing run.
- **Energy per token** is about unchanged with one request, and up to 17% higher with eight, because the GPU draws more power while it runs faster.
- **Batch-invariant mode.** With `VLLM_BATCH_INVARIANT=1`, every layer uses vLLM's standard path.

## Tests

```bash
pip install pytest
pytest -s
```

Run them from the repository root after installing. The GPU tests check:

- every kernel against an exact FP32 reference on Qwen3-8B's layer shapes;
- that a `torch.compile` graph still picks the kernel by the real batch size;
- that the split-K kernel gives identical results run to run.

They are skipped without a GPU. `tests/test_archs.py` needs no GPU.

## Benchmark

```bash
python benchmarks/kernel_sweep.py --models qwen3-8b --trials 5 > sweep.jsonl
```

This is the single-kernel benchmark from the paper.

- **Kernels:** cuBLAS BF16, vLLM's CUTLASS FP8 with and without its input conversion, FP8 Marlin, and ours.
- **Shapes:** the layers of Qwen3-8B, Qwen3.8-27B and Llama-3.3-70B, at 1 to 64 tokens.
- **Memory traffic:** weights rotate through more copies than fit in L2 cache, so every call reads them from memory, as in a real model.
- **Output:** one JSON object per line. Lines with `"kind": "time"` hold microseconds per call.

## Using the kernels directly

From PyTorch, import torch before the extension:

```python
import torch
from sr_vllm import _C

x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)  # [M, K]
w = (torch.randn(6144, 4096, device="cuda") * 0.02).to(torch.float8_e4m3fn)  # [N, K]
scale = torch.full((6144,), 1.0, device="cuda")  # fp32, one per output row
y = _C.gemv_fp8w(x, w, scale, 0)  # (x @ w.T) * scale, bf16 [M, N]
```

| Function | Tokens (M) | K must be a multiple of | Last argument |
|---|---|---|---|
| `gemv_fp8w(x, w, scale, variant)` | 1 to 8 | 16 | tile variant, `0` to `num_variants - 1` |
| `tc_gemv_fp8w(x, w, scale, variant)` | 1 to 64 | 64 | tile variant, `0` to `tc_num_variants - 1` |
| `gemv_bf16(x, w, variant)` and `tc_gemv_bf16(x, w, variant)` | as above | 8 and 32 | BF16 weights instead of FP8 |

From C++ or CUDA, include the headers in `kernels/gemv/`. They need nothing but the CUDA toolkit. The entry points are:

- `sr::gemv::gemv_fp8w`
- `sr::gemv_tc::gemv_fp8w`
- `sr::gemm_tc_wide::gemm_fp8w`

Each returns a `cudaError_t`, and each header states its requirements above the function.

## Repository layout

```
kernels/gemv/   CUDA kernels (header-only) and the PyTorch binding
src/sr_vllm/    the vLLM plugin
tests/          GPU and CPU tests
benchmarks/     the single-kernel benchmark
```

The `sr_` in names such as `sr_w8a16` is short for SuperRabbit, F0 Labs' inference platform, which these kernels were built for.

## Citation

```bibtex
@techreport{mohanty2026weightonlyfp8,
  title       = {When Quantizing Activations Costs More Than It Saves: Weight-Only {FP8} Kernels for Faster {LLM} Decoding},
  author      = {Mohanty, Suvendu},
  institution = {F0 Labs},
  year        = {2026}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
