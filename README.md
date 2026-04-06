# 🚀 VLM Research Bench UI (Simple Edition)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA: 13.0](https://img.shields.io/badge/CUDA-13.0-blue.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Blackwell: Optimized](https://img.shields.io/badge/Blackwell-SM_121-green.svg)](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)

This repository is a **VLM (Vision Language Model) Inference Performance Benchmark Tool** optimized for the **NVIDIA Blackwell (DGX Spark)** architecture. It aims to serve 7 state-of-the-art VLM models simultaneously and calculate precision TPS (Tokens Per Second) through rigorous repeated measurements.

---

## ✨ Key Features (English)
- **7-Model Simultaneous Serving**: Run Qwen 3.5 (2B, 35B), InternVL 3.5, LFM 2.5, Next2 Air, and Gemma 4 (26B, E2B) on individual ports (**8081-8087**).
- **Professional Benchmark Engine**: Conduct 20 consecutive inference runs and calculate **Average TPS over 19 runs**, excluding the first warmup data.
- **Blackwell Optimization**: Native utilization of CUDA 13.0 and SM121 hardware acceleration (Flash Attention, Native FP4, etc.) for peak GGUF inference.

## 📦 Getting Started (Docker)

### 1. Prerequisites
- **GPU**: NVIDIA Blackwell (SM121) Recommended. (Other GPUs require modifying `CMAKE_CUDA_ARCHITECTURES` in `Dockerfile`)
- **VRAM**: 100GB+ VRAM required for full model operation (Optimized for GB10 124GB).
- **Model Files**: Place GGUF and mmproj files in the `models/` folder. (See [MODEL_CREDITS.md](./MODEL_CREDITS.md) for details)

### 2. Run (Using Pre-built Image)
```bash
# Pull the latest image
docker pull ghcr.io/sowilow/dgx-spark-llama.cpp-bench:latest

# Run (All model servers start automatically)
docker-compose up -d
```

---

# 🚀 VLM Research Bench UI (간편 버전)

이 레포지토리는 **NVIDIA Blackwell (DGX Spark)** 아키텍처에 최적화된 **VLM(Vision Language Model) 추론 성능 벤치마크 도구**입니다. 7종의 최신 VLM 모델을 동시에 구동하고, 정밀한 반복 측정을 통해 실제 TPS(Tokens Per Second)를 산출하는 데 목적이 있습니다.

---

## ✨ 주요 특징 (한국어)
- **7종 모델 동시 서빙**: Qwen 3.5 (2B, 35B), InternVL 3.5, LFM 2.5, Next2 Air, 그리고 Gemma 4 (26B, E2B)를 개별 포트(**8081-8087**)에서 동시에 구동.
- **전문 벤치마크 엔진**: 20회 연속 추론 수행 및 첫 번째 웜업(Warmup) 데이터를 제외한 **19회 평균 TPS** 산출.
- **Blackwell 최적화**: CUDA 13.0 및 SM121 하드웨어 가속(Flash Attention, Native FP4 등)을 활용한 GGUF 추론 최대로 수행.

## 📦 설치 및 구동 (Docker)

### 1. 전제 조건 (Prerequisites)
- **GPU**: NVIDIA Blackwell (SM121) 권장. (기타 GPU는 `Dockerfile`의 `CMAKE_CUDA_ARCHITECTURES` 수정 필요)
- **VRAM**: 전체 모델 가동 시 최소 100GB+ VRAM 필요 (GB10 124GB 최적)
- **Model Files**: 각 모델의 GGUF 및 mmproj 파일을 `models/` 폴더에 배치하십시오. (상세 내역은 [MODEL_CREDITS.md](./MODEL_CREDITS.md) 참조)

### 2. 실행 (사전 빌드된 이미지 사용)
```bash
# 이미지 내려받기
docker pull ghcr.io/sowilow/dgx-spark-llama.cpp-bench:latest

# 실행 (기본적으로 모든 모델 서버가 자동 시작됨)
docker-compose up -d
```

---

## ⚖️ License & Disclaimer / 라이선스 및 면책 조항
1. **Software License**: The source code is under **MIT License**.
2. **Model Weights License**: Individual model weights follow their respective licenses (Google, Alibaba, OpenGVLab, Liquid AI, etc.).
3. **Liability Disclaimer**: This tool is for performance measurement and research purposes only. Users are responsible for legal compliance regarding model outputs and commercial usage.

---
*Developed and Optimized on NVIDIA Blackwell (DGX Spark) by sowilow*
