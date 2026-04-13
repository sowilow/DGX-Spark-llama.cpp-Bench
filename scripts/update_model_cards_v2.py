# scripts/update_model_cards_v2.py
import os
import sys
from huggingface_hub import HfApi, hf_hub_download, upload_file

PROMO_TAG = "## 🚀 v0.1.6: Real-time Metrics & Blackwell-Optimized Docker"
PROMO_CONTENT = """

---

## 🚀 v0.1.6: Real-time Metrics & Blackwell-Optimized Docker (Recommended)

This model is fully compatible with the **[DGX-Spark-llama.cpp-Bench](https://github.com/sowilow/DGX-Spark-llama.cpp-Bench)**.
Experience the state-of-the-art inference engine optimized for NVIDIA Blackwell (DGX Spark) hardware.

### 🌟 Key Features (v0.1.6)
- **Real-time Performance Metrics**: Now visualizes `Input TPS` and `Output TPS` during streaming.
- **Improved Reasoning UI**: Seamlessly renders and stabilizes the model's Chain-of-Thought (CoT).
- **Blackwell Optimization**: Native support for ARM64/SM121 and CUDA 13.0 FP4.

### 🐳 Quick Start
```bash
# Pull the latest optimized image
docker pull ghcr.io/sowilow/dgx-spark-llama.cpp-bench:v0.1.6
```
For more details, visit our [GitHub Repository](https://github.com/sowilow/DGX-Spark-llama.cpp-Bench).

---

## 🚀 v0.1.6: 실시간 지표 및 Blackwell 최적화 도커 (권장)

이 모델은 **[DGX-Spark-llama.cpp-Bench](https://github.com/sowilow/DGX-Spark-llama.cpp-Bench)** 시스템에 최적화되어 있습니다.
NVIDIA Blackwell (DGX Spark) 하드웨어의 성능을 최대로 활용하세요.

### 🌟 주요 특징 (v0.1.6)
- **실시간 성능 지표 시각화**: 스트리밍 중 `Input TPS` 및 `Output TPS`를 실시간으로 표시합니다.
- **지능형 추론 UI 고도화**: 모델의 생각하는 과정(CoT)을 더 안정적으로 렌더링합니다.
- **Blackwell 최적화**: ARM64/SM121 아키텍처 및 CUDA 13.0 FP4 가속 지원.

### 🐳 실행 방법
```bash
# 최신 최적화 이미지 내려받기
docker pull ghcr.io/sowilow/dgx-spark-llama.cpp-bench:v0.1.6
```
상세한 사용법은 [GitHub 리포지토리](https://github.com/sowilow/DGX-Spark-llama.cpp-Bench)를 참조하세요.

---
"""

def update_model_card(repo_id, api, token):
    print(f"--- Processing {repo_id} ---", flush=True)
    try:
        # Download README.md
        print(f"  Downloading README.md...", flush=True)
        try:
            readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", token=token, force_download=True)
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
            print(f"  Download successful ({len(content)} bytes)", flush=True)
        except Exception as e:
            print(f"  [INFO] README.md not found or error ({e}), creating new one.", flush=True)
            content = ""
        
        if PROMO_TAG in content:
            print(f"  [SKIPPED] Promo already exists.", flush=True)
            return

        # Insert promo content
        print(f"  Preparing new content...", flush=True)
        parts = content.split("---")
        if len(parts) >= 3 and content.strip().startswith("---"):
            new_content = "---" + parts[1] + "---" + PROMO_CONTENT + "---".join(parts[2:])
        else:
            new_content = PROMO_CONTENT + content
            
        # Save temp file
        temp_readme = f"temp_README_{repo_id.replace('/', '_')}.md"
        with open(temp_readme, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        # Upload
        print(f"  Uploading to Hugging Face...", flush=True)
        api.upload_file(
            path_or_fileobj=temp_readme,
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message="Update README with Docker Quick Start info"
        )
        if os.path.exists(temp_readme):
            os.remove(temp_readme)
        print(f"  [SUCCESS] Updated {repo_id}", flush=True)
        
    except Exception as e:
        print(f"  [ERROR] Failed to update {repo_id}: {e}", flush=True)

def main():
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        print("HF_TOKEN environment variable not found.", flush=True)
        sys.exit(1)
        
    api = HfApi(token=api_token)
    models = list(api.list_models(author="sowilow"))
    print(f"Found {len(models)} models.", flush=True)
    
    for m in models:
        update_model_card(m.modelId, api, api_token)

if __name__ == "__main__":
    main()
