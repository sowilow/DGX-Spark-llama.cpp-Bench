# 🚀 VLM Research Bench UI (Simple Edition)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA: 13.0](https://img.shields.io/badge/CUDA-13.0-blue.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Blackwell: Optimized](https://img.shields.io/badge/Blackwell-SM_121-green.svg)](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)

이 레포지토리는 **NVIDIA Blackwell (DGX Spark)** 아키텍처에 최적화된 **VLM(Vision Language Model) 추론 성능 벤치마크 도구**입니다. 5종의 최신 VLM 모델을 동시에 구동하고, 정밀한 반복 측정을 통해 실제 TPS(Tokens Per Second)를 산출하는 데 목적이 있습니다.

---

## ✨ 주요 특징
- **5종 모델 동시 서빙**: Qwen 2.5(2B, 35B), InternVL 3.5, LFM 2.5, Next2 Air를 개별 포트(8081-8085)에서 동시에 구동.
- **전문 벤치마크 엔진**: 20회 연속 추론 수행 및 첫 번째 웜업(Warmup) 데이터를 제외한 **19회 평균 TPS** 산출.
- **Blackwell 최적화**: CUDA 13.0 및 SM121 하드웨어 가속(Flash Attention, Native FP4 등)을 활용한 GGUF 추론 최대로 수행.

## 📦 설치 및 구동 (Docker)

### 1. 전제 조건 (Prerequisites)
- **GPU**: NVIDIA Blackwell (SM121) 권장 (기타 GPU는 `Dockerfile`의 `CMAKE_CUDA_ARCHITECTURES` 수정 필요)
- **VRAM**: 전체 모델 가동 시 최소 100GB+ VRAM 필요 (GB10 124GB 최적)
- **Model Files**: 각 모델의 GGUF 및 mmproj 파일을 `models/` 폴더에 배치하십시오. (상세 정보는 [MODEL_CREDITS.md](./MODEL_CREDITS.md) 참조)

### 2. 실행
```bash
# 컨테이너 빌드 및 백그라운드 실행
docker-compose up -d --build
```
UI 접속: `http://localhost:7860`

---

## ⚖️ 라이선스 및 면책 조항 (License & Disclaimer)

1. **Software License**: 본 프로젝트의 소스 코드는 **MIT License**를 따릅니다.
2. **Model Weights License**: 구동되는 각 모델 가중치는 원저작자의 개별 라이선스(Apache 2.0, Qwen, Liquid AI 등)를 따릅니다.
3. **Liability Disclaimer**: 본 도구는 성능 측정 및 연구 목적으로만 제공되며, 모델 출력물에 의한 법적 책임이나 상업적 이용에 따른 라이선스 위반 책임은 사용자에게 있습니다.

상세 내용은 [LICENSE](./LICENSE) 파일을 확인해 주십시오.

## 🤝 기여 (Attribution)
본 프로젝트는 **NVIDIA DGX Spark** 개발자 커뮤니티와 여러 VLM 연구 그룹의 성과물을 기반으로 제작되었습니다.
- [MODEL_CREDITS.md](./MODEL_CREDITS.md): 각 모델에 대한 상세 정보
