# 🚀 VLM Research Bench UI (Simple Edition)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA: 13.0](https://img.shields.io/badge/CUDA-13.0-blue.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Blackwell: Optimized](https://img.shields.io/badge/Blackwell-SM_121-green.svg)](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)

This repository is a **VLM (Vision Language Model) Inference Performance Benchmark Tool** optimized for the **NVIDIA Blackwell (DGX Spark)** architecture. It supports simultaneous serving of multiple state-of-the-art VLM models and calculates precise TPS (Tokens Per Second).

---

## ✨ Key Features (v0.1.5)
- **Blackwell (DGX Spark) Optimization**: Enhanced performance for ARM64/SM121 with CUDA 13.0 and native FP4 support.
- **Real-time Performance Metrics**: Visualizes `Input TPS` and `Output TPS` during streaming for precise benchmarking.
- **Intelligent Reasoning UI**:
    - **Dynamic Detection**: Automatically detects and enables reasoning mode for models like **Gemma-4**.
    - **Thought Visualization**: Professional Markdown blocks for `reasoning_content` in Playground.
- **Real-time Streaming Output**: Stable character-by-character generation with Svelte 5 / Gradio 6 support.
- **Multi-Model Simultaneous Serving**: Run Qwen 3.5, InternVL 3.5, LFM 2.5, Next2 Air, Gemma 4, and **OpenAI GPT-OSS 20B** on individual ports.
- **Unified Port Mapping**: 
    - **7860**: Gradio Web UI
    - **8100-8150**: Pre-allocated Model Server Ports (Categorized by family)
- **Blackwell Optimization**: Native utilization of CUDA 13.0 and SM121 hardware acceleration (Flash Attention, Native FP4/MXFP4, etc.).

## 📦 Getting Started (Docker)

### 1. Prerequisites
- **GPU**: NVIDIA Blackwell (SM121) Recommended.
- **VRAM**: 100GB+ VRAM required for full model operation.
- **Model Files**: Place GGUF and mmproj files in the `models/` folder.

### 2. Run (Using Pre-built Image)
```bash
# Pull the latest image
docker pull ghcr.io/sowilow/dgx-spark-llama.cpp-bench:v0.1.5

# Run (UI available at http://localhost:7860)
docker compose up -d
```

---

# 🚀 VLM Research Bench UI (간편 버전)

이 레포지토리는 **NVIDIA Blackwell (DGX Spark)** 아키텍처에 최적화된 **VLM(Vision Language Model) 추론 성능 벤치마크 도구**입니다. 여러 종류의 최신 VLM 모델을 동시에 구동하고, 정밀한 반복 측정을 통해 실제 TPS(Tokens Per Second)를 산출합니다.

---

## ✨ 주요 특징 (v0.1.5)
- **Blackwell (DGX Spark) 최적화**: ARM64/SM121 환경 최적화 및 CUDA 13.0 FP4 하드웨어 가속 활용.
- **실시간 성능 지표 시각화**: 스트리밍 중 `Input TPS` 및 `Output TPS`를 실시간으로 표시합니다.
- **지능형 추론 가시화 (Reasoning UI)**:
    - **자동 감지**: **Gemma-4** 등 추론 지원 모델을 자동으로 인식하여 전용 UI 옵션을 활성화합니다.
    - **과정 시각화**: 모델의 생각하는 과정(`reasoning_content`)을 깔끔한 Markdown 블록으로 표시합니다.
- **실시간 스트리밍 출력**: Gradio 6 / Svelte 5 기반의 끊김 없는 실시간 답변 생성.
- **복수 모델 동시 서빙**: Qwen 3.5, InternVL 3.5, LFM 2.5, Next2 Air, Gemma 4, 그리고 **OpenAI GPT-OSS 20B**를 개별 포트에서 동시에 구동 가능.
- **통합 포트 매핑**:
    - **7860**: Gradio 웹 인터페이스 (UI)
    - **8100-8150**: 모델 서버 전용 포트 (제품군별 그룹화)
- **Blackwell 최적화**: CUDA 13.0 및 SM121 하드웨어 가속(Flash Attention, Native FP4/MXFP4 등) 활용.

## 📦 설치 및 구동 (Docker)

### 1. 전제 조건 (Prerequisites)
- **GPU**: NVIDIA Blackwell (SM121) 권장.
- **VRAM**: 전체 모델 가동 시 최소 100GB+ VRAM 필요.
- **Model Files**: 각 모델의 GGUF 및 mmproj 파일을 `models/` 폴더에 배치하십시오.

### 2. 실행 (사전 빌드된 이미지 사용)
```bash
# 이미지 내려받기
docker pull ghcr.io/sowilow/dgx-spark-llama.cpp-bench:v0.1.5

# 실행 (UI 접속 주소: http://localhost:7860)
docker compose up -d
```

---

## ⚖️ License & Disclaimer / 라이선스 및 면책 조항
1. **Software License**: The source code is under **MIT License**.
2. **Model Weights License**: Individual model weights follow their respective licenses.
3. **Liability Disclaimer**: This tool is for performance measurement and research purposes only.

---
*Developed and Optimized on NVIDIA Blackwell (DGX Spark) by sowilow*
