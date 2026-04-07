# scripts/update_hf_tags.py
import os
import yaml
from huggingface_hub import HfApi, hf_hub_download

def get_bits_from_files(files):
    bits = set()
    for f in files:
        low = f.lower()
        if any(x in low for x in ["q4", "mxfp4", "4bit"]): bits.add("4-bit")
        if any(x in low for x in ["q8", "8bit"]): bits.add("8-bit")
    return list(bits)

def update_repo_tags(repo_id, api, token):
    print(f"--- Processing {repo_id} ---")
    try:
        # 1. Determine bits from files
        repo_files = api.list_repo_files(repo_id=repo_id, token=token)
        tags_to_add = get_bits_from_files(repo_files)
        tags_to_add.append("dgx-spark")
        
        # 2. Download README.md
        try:
            readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", token=token, force_download=True)
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
        except:
            content = "---\ntags: []\n---\n"

        # 3. Parse and Update
        parts = content.split("---")
        if len(parts) >= 3 and content.strip().startswith("---"):
            fm_text = parts[1]
            body_text = "---".join(parts[2:])
            
            try:
                fm = yaml.safe_load(fm_text) or {}
            except Exception as e:
                print(f"  [Error] YAML parse failed: {e}")
                return

            existing_tags = fm.get("tags", [])
            if isinstance(existing_tags, str): existing_tags = [existing_tags]
            elif existing_tags is None: existing_tags = []
            
            new_tags = sorted(list(set([str(t) for t in (existing_tags + tags_to_add) if str(t).strip()])))
            
            if set(new_tags) == set(existing_tags):
                print(f"  [Skipped] Tags already set: {new_tags}")
                return
            
            fm["tags"] = new_tags
            new_fm_text = yaml.dump(fm, allow_unicode=True, sort_keys=False)
            updated_content = "---\n" + new_fm_text + "---\n" + body_text
        else:
            # No frontmatter
            new_tags = sorted(list(set(tags_to_add)))
            updated_content = "---\ntags: [" + ", ".join(new_tags) + "]\n---\n" + content

        # 4. Upload
        temp_file = f"temp_readme_{repo_id.replace('/', '_')}.md"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(updated_content)
        
        api.upload_file(
            path_or_fileobj=temp_file,
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message=f"Update tags: {', '.join(tags_to_add)}"
        )
        if os.path.exists(temp_file): os.remove(temp_file)
        print(f"  [Success] Updated {repo_id} with {new_tags}")

    except Exception as e:
        print(f"  [Error] {e}")

if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    models = api.list_models(author="sowilow")
    for m in models:
        update_repo_tags(m.modelId, api, token)
