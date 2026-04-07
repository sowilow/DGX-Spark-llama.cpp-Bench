# scripts/update_model_cards.py
import os
import sys
from huggingface_hub import HfApi, ModelCard

PROMO_TAG = "## 🚀 Quick Start with Docker (Recommended)"
PROMO_CONTENT = """
---

## 🚀 Quick Start with Docker (Recommended)

You can easily run this model using the **DGX-Spark-llama.cpp-Bench** inference engine. It's pre-configured for high-performance inference on NVIDIA hardware (especially Blackwell/DGX Spark).

### 1. Pull the Docker Image
```bash
docker pull ghcr.io/sowilow/dgx-spark-llama.cpp-bench:latest
```

### 2. Run the Inference Server
For detailed configuration and usage, visit the [GitHub Repository](https://github.com/sowilow/DGX-Spark-llama.cpp-Bench).

---
"""

def update_model_card(repo_id, api):
    print(f"Checking {repo_id}...")
    try:
        # Download README.md content
        card = ModelCard.load(repo_id)
        content = card.text
        
        if PROMO_TAG in content:
            print(f"  [SKIPPED] Promo already exists in {repo_id}")
            return

        # Insert promo content after YAML frontmatter
        # ModelCard.load already handles splitting the metadata and text
        # But we want to preserve the metadata if possible by using card.push_to_hub
        
        # Strategy: Prepend to the top of the body text
        new_content = PROMO_CONTENT + "\n" + content
        card.text = new_content
        
        # Push back
        card.push_to_hub(repo_id, commit_message="Update README with Docker Quick Start info")
        print(f"  [SUCCESS] Updated {repo_id}")
        
    except Exception as e:
        print(f"  [ERROR] Failed to update {repo_id}: {e}")

def main():
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        print("HF_TOKEN environment variable not found.")
        sys.exit(1)
        
    api = HfApi(token=api_token)
    models = api.list_models(author="sowilow")
    
    for m in models:
        update_model_card(m.modelId, api)

if __name__ == "__main__":
    main()
