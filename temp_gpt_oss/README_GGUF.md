---
license: apache-2.0
pipeline_tag: text-generation
library_name: gguf
tags:
- llama-cpp
- gguf
- MoE
- quantization
- DGX-Spark
- Blackwell
- nvidia
- gpt-oss
base_model: openai/gpt-oss-20b
---

# gpt-oss-20b-DGX-Spark-GGUF

<p align="center">
  <img alt="gpt-oss-20b" src="https://raw.githubusercontent.com/openai/gpt-oss/main/docs/gpt-oss-20b.svg">
</p>

This repository provides **GGUF** quantized versions of [OpenAI's gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b), optimized specifically for **NVIDIA Blackwell (DGX Spark)** architectures.

These models were converted and quantized using [llama.cpp](https://github.com/ggml-org/llama.cpp) with support for the `gpt_oss` architecture.

## Model Highlights

*   **Optimized for Blackwell**: Specifically tuned for high-performance inference on NVIDIA DGX Spark (SM120/SM121).
*   **Flexible Quantization**: 
    *   `Q4_K_M`: 4-bit Medium quantization (recommended for efficiency).
    *   `Q8_0`: 8-bit quantization (recommended for maximum precision).
*   **MoE Architecture**: 21B total parameters with 3.6B active parameters, leveraging Mixture-of-Experts for high efficiency.
*   **Long Context**: Supports up to 131k context length.

## Quantization Details

| File | Quant Method | Bitrate | Size | Description |
|------|--------------|---------|------|-------------|
| `gpt-oss-20b-q4_k_m.gguf` | Q4_K_M | 4.5 bpw | ~12 GB | Balanced performance and quality. |
| `gpt-oss-20b-q8_0.gguf` | Q8_0 | 8.5 bpw | ~22 GB | Standard 8-bit quantization. |

## Quick Start (llama.cpp)

To run these models on a DGX Spark system:

1.  **Pull the optimized Docker image**:
    ```bash
    docker pull ghcr.io/sowilow/dgx-spark-llama.cpp-bench:latest
    ```

2.  **Run with llama-server**:
    ```bash
    docker run --gpus all -v $(pwd)/models:/model \
        ghcr.io/sowilow/dgx-spark-llama.cpp-bench:latest \
        llama-server -m /model/gpt-oss-20b-q4_k_m.gguf -ngl 99 -c 8192
    ```

## Original Model Information

This is a quantized version of [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b). 
Please refer to the original model card for details on training, safety, and benchmarks.

### Citation
```bibtex
@misc{openai2025gptoss120bgptoss20bmodel,
      title={gpt-oss-120b & gpt-oss-20b Model Card}, 
      author={OpenAI},
      year={2025},
      eprint={2508.10925},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.10925}, 
}
```
