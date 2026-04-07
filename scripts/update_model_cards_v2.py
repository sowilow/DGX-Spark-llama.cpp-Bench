# scripts/update_model_cards_v2.py
import os
import sys
from huggingface_hub import HfApi, hf_hub_download, upload_file

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
