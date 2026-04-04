---
license: gemma
base_model: google/gemma-4-26B-A4B-it
language:
- en
- ko
pipeline_tag: image-text-to-text
tags:
- vlm
- gguf
- gemma
- blackwell-optimized
- sm121
- quantized
- moe
---

# gemma-4-26b-a4b-it-GGUF

This repository contains GGUF-quantized weights for **Gemma-4-26B-A4B-it**, specifically optimized for **NVIDIA Blackwell (DGX Spark)** hardware.

## 🚀 Key Features
- **Hardware Optimized**: Built with CUDA 13.0 and SM121 (Blackwell) native acceleration.
- **Quantization**: Q4_K_M (4-bit unified quantization) for balanced performance and accuracy.
- **MoE Architecture**: Fully optimized MoE routing for high-throughput inference on GB10.
- **Base Model Integration**: Linked directly to the original [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it).

## ⚖️ License & Attribution
This model is a quantized version of the original [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) and is subject to the **Gemma License Agreement**.

## 📂 Files Included
- `gemma-4-26b-a4b-it-q4_k_m.gguf`: Main MoE model weights.
- `gemma-4-26b-vision-mmproj-f16.gguf`: Multimodal vision projector (Dimension-matched: n_embd=2816).

---
*Created using [DGX-Spark-llama.cpp-Bench](https://github.com/sowilow/DGX-Spark-llama.cpp-Bench)*
