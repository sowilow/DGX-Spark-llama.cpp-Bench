import os
from huggingface_hub import hf_hub_download, snapshot_download

repo_id = "Qwen/Qwen3.5-35B-A3B"
local_dir = "models/Qwen3.5-35B-A3B-hf"
os.makedirs(local_dir, exist_ok=True)

# 필수 설정 파일
files = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt"
]

for f in files:
    print(f"Downloading {f}...")
    hf_hub_download(repo_id=repo_id, filename=f, local_dir=local_dir)

# Shard 1 다운로드 (비전 가중치 포함 가능성 99%)
shard_1 = "model.safetensors-00001-of-00014.safetensors"
print(f"Downloading {shard_1}...")
hf_hub_download(repo_id=repo_id, filename=shard_1, local_dir=local_dir)

print("Download complete.")
