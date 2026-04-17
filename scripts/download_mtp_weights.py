#!/usr/bin/env python3
"""
Download MTP (Multi-Token Prediction) weights from Intel AutoRound checkpoint.

We only need:
  - model_extra_tensors.safetensors  (MTP head weights, ~4-5GB)
  - model.safetensors.index.json     (tensor name mapping)
  - config.json, tokenizer*.json     (config & tokenizer for conversion)

Full model is 11 shards (~22GB) → NOT needed since we only want the MTP head.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

REPO_ID = "Intel/Qwen3.5-35B-A3B-int4-AutoRound"
TARGET_DIR = Path("models/Intel-Qwen3.5-35B-A3B-MTP")

# Files to selectively download (skip 11 model shards = ~22GB savings)
NEEDED_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model_extra_tensors.safetensors",   # ← MTP Head weights
    "tokenizer.json",
    "tokenizer_config.json",
    "quantization_config.json",
]

def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    api = HfApi()
    
    print(f"Downloading selected files from {REPO_ID}")
    print(f"Target: {TARGET_DIR.resolve()}")
    print("=" * 60)
    
    for filename in NEEDED_FILES:
        target_path = TARGET_DIR / filename
        if target_path.exists():
            size_gb = target_path.stat().st_size / 1e9
            print(f"[SKIP]  {filename} (already exists, {size_gb:.2f} GB)")
            continue
        
        print(f"[DOWN]  {filename} ...")
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=str(TARGET_DIR),
                local_dir_use_symlinks=False,
            )
            size_gb = Path(local_path).stat().st_size / 1e9
            print(f"        → Done ({size_gb:.2f} GB)")
        except Exception as e:
            print(f"        → ERROR: {e}", file=sys.stderr)
            sys.exit(1)
    
    print("=" * 60)
    print("Download complete. Now run the MTP extraction:")
    print(f"  python3 scripts/extract_mtp_draft.py --source {TARGET_DIR} --output models/qwen3.5-35b-mtp-draft.gguf")

if __name__ == "__main__":
    main()
