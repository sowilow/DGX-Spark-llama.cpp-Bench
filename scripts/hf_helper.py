# scripts/hf_helper.py
import sys
import os
from huggingface_hub import hf_hub_download, HfApi

def download_model(repo_id, local_dir):
    print(f"Downloading {repo_id} to {local_dir}...")
    # Base model weights (safetensors)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.bin", "*.pth", "*.msgpack"] # Prefer safetensors
        )
        print(f"Successfully downloaded {repo_id}")
    except Exception as e:
        print(f"Error downloading {repo_id}: {e}")
        sys.exit(1)

def upload_file(repo_id, local_file, path_in_repo, token=None):
    api = HfApi(token=token)
    print(f"Uploading {local_file} to {repo_id}/{path_in_repo}...")
    try:
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"Successfully uploaded {path_in_repo}")
    except Exception as e:
        print(f"Error uploading to {repo_id}: {e}")

def create_repository(repo_id, token=None):
    api = HfApi(token=token)
    print(f"Creating repository {repo_id}...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"Repository {repo_id} is ready.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        # Don't exit here, maybe next file works

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 hf_helper.py [download|upload] [repo_id] [path]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    repo = sys.argv[2]
    path = sys.argv[3]
    token = os.environ.get("HF_TOKEN")
    
    if cmd == "download":
        download_model(repo, path)
    elif cmd == "upload":
        if len(sys.argv) < 5:
             print("Usage: python3 hf_helper.py upload [repo_id] [local_path] [repo_path]")
             sys.exit(1)
        upload_file(repo, path, sys.argv[4], token=token)
    elif cmd == "create":
        create_repository(repo, token=token)
