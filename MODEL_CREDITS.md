# Model Credits & Attributions

본 프로젝트는 최첨단 시각 언어 모델(VLM)의 성능 벤치마크를 위해 아래 연구 그룹 및 조직의 성과물을 활용하고 있습니다. 원저작자들의 공헌에 깊은 감사를 표하며, 각 모델의 라이선스를 준수하여 배포하고 있습니다.

| Library Name | Version | Original Organizations | License |
| :--- | :--- | :--- | :--- |
| **Qwen 3.5 (2B/35B)** | [Qwen3.5](https://github.com/QwenLM/Qwen2.5) | Alibaba Cloud / Qwen | Qwen License |
| **InternVL 3.5** | [InternVL3.5](https://github.com/OpenGVLab/InternVL) | OpenGVLab | Apache 2.0 |
| **LFM 2.5 (VL)** | [LFM-1.6B-VL](https://liquid.ai) | Liquid AI | Liquid Community |
| **Next2-Air** | [Next-2-Air](https://huggingface.co/thelamapi/Next-2-Air-GGUF) | thelamapi / Next-2 | Apache 2.0 |
| **Gemma 4 (26B/E2B)** | [Gemma-4](https://github.com/google-deepmind/gemma) | Google DeepMind | Apache 2.0 |
| **GPT-OSS 20B** | [GPT-OSS](https://huggingface.co/openai/gpt-oss-20b) | OpenAI | Apache 2.0 |

---

### 📂 Optimized Resources (by sowilow)

NVIDIA Blackwell (DGX Spark) 하드웨어에 최적화된 GGUF 가중치는 아래 공식 저장소에서 다운로드하실 수 있습니다.

- **Qwen 3.5 35B-GGUF**: [sowilow/Qwen3.5-35B-A3B-DGX-Spark-GGUF](https://huggingface.co/sowilow/Qwen3.5-35B-A3B-DGX-Spark-GGUF)
- **Qwen 3.5 2B-GGUF**: [sowilow/Qwen3.5-2B-DGX-Spark-GGUF](https://huggingface.co/sowilow/Qwen3.5-2B-DGX-Spark-GGUF)
- **LFM 2.5 1.6B-GGUF**: [sowilow/LFM-2.5-1.6B-DGX-Spark-GGUF](https://huggingface.co/sowilow/LFM-2.5-1.6B-DGX-Spark-GGUF)
- **InternVL 3.5 2B-GGUF**: [sowilow/InternVL-3.5-2B-DGX-Spark-GGUF](https://huggingface.co/sowilow/InternVL-3.5-2B-DGX-Spark-GGUF)
- **Next2-Air-GGUF**: [sowilow/Next2-Air-DGX-Spark-GGUF](https://huggingface.co/sowilow/Next2-Air-DGX-Spark-GGUF)
- **Gemma 4 26B-GGUF**: [sowilow/gemma-4-26b-a4b-it-DGX-Spark-GGUF](https://huggingface.co/sowilow/gemma-4-26b-a4b-it-DGX-Spark-GGUF)
- **Gemma 4 E2B-GGUF**: [sowilow/gemma-4-e2b-it-DGX-Spark-GGUF](https://huggingface.co/sowilow/gemma-4-e2b-it-DGX-Spark-GGUF)
- **GPT-OSS 20B-GGUF**: [sowilow/gpt-oss-20b-DGX-Spark-GGUF](https://huggingface.co/sowilow/gpt-oss-20b-DGX-Spark-GGUF)

---

> [!NOTE]
> All GGUF conversions are proprietary optimized for NVIDIA Blackwell(SM121) architecture to achieve maximum TPS (Tokens Per Second) performance.
