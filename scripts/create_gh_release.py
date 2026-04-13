import os
import requests
import json
import sys

def create_release():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN environment variable not found.")
        sys.exit(1)
        
    owner = "sowilow"
    repo = "DGX-Spark-llama.cpp-Bench"
    tag_name = "v0.1.6"
    
    url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {
        "tag_name": tag_name,
        "name": f"Release {tag_name}",
        "body": "## 🚀 v0.1.6 Release\n\n- **Gradio 6 Support**: Fixed image upload and multimodal interaction.\n- **Real-time Metrics**: Integrated native llama.cpp timings for accurate TPS display.\n- **Model Registry Sync**: Updated all model links to current repositories.\n- **Blackwell Optimization**: Performance improvements for ARM64/SM121.",
        "draft": False,
        "prerelease": False,
        "generate_release_notes": False
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 201:
        print(f"Successfully created release: {response.json()['html_url']}")
    elif response.status_code == 422: # Unprocessable Entity (e.g. release already exists)
        print(f"Release already exists or validation failed: {response.status_code}")
        print(response.text)
    else:
        print(f"Failed to create release: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    create_release()
