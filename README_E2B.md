---
license: gemma
base_model: google/gemma-4-E2B-it
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
---

# gemma-4-e2b-it-GGUF

This repository contains GGUF-quantized weights for **Gemma-4-E2B-it**, specifically optimized for **NVIDIA Blackwell (DGX Spark)** hardware.

## 🚀 Key Features
- **Hardware Optimized**: Built with CUDA 13.0 and SM121 (Blackwell) native acceleration.
- **Quantization**: Q4_K_M (4-bit unified quantization) for ultra-low latency vision tasks.
- **Base Model Integration**: Linked directly to the original [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it).

## ⚖️ License & Attribution
This model is a quantized version of the original [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) and is subject to the **Gemma License Agreement**.

## 📂 Files Included
- `gemma-4-e2b-it-q4_k_m.gguf`: Main model weights.
- `gemma-4-e2b-vision-mmproj-f16.gguf`: Multimodal vision projector (Dimension-matched: n_embd=1536).

---
*Created using [DGX-Spark-llama.cpp-Bench](https://github.com/sowilow/DGX-Spark-llama.cpp-Bench)*
