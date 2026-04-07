# bench_UI_simple/scripts/test_reasoning.py
import requests
import json
import time
import os
import subprocess
import yaml

def test_model_reasoning(model_name, port, model_path, mmproj_path=None):
    results = {}
    for case in ["off", "on"]:
        print(f"\n--- Testing {model_name} (Reasoning: {case}) ---")
        
        # Kill existing llama-server on this port
        subprocess.run(f"fuser -k {port}/tcp", shell=True, stderr=subprocess.DEVNULL)
        time.sleep(2)

        # Build command
        cmd = [
            "/usr/local/bin/llama-server",
            "--host", "0.0.0.0",
            "-m", model_path,
            "--port", str(port),
            "-ngl", "999",
            "--ctx-size", "2048",
            "--flash-attn", "on",
            "--reasoning", case
        ]
        if mmproj_path and os.path.exists(mmproj_path):
            cmd.extend(["--mmproj", mmproj_path])

        log_file = f"test_reasoning_{model_name}_{case}.log"
        with open(log_file, "w") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=f)

        # Wait for ready
        ready = False
        for i in range(30):
            try:
                resp = requests.get(f"http://localhost:{port}/health", timeout=1)
                if resp.status_code == 200:
                    ready = True
                    break
            except:
                pass
            time.sleep(1)

        if not ready:
            print(f"Failed to start server for {model_name} ({case})")
            proc.terminate()
            results[case] = "Server Start Failed"
            continue

        # Send query
        payload = {
            "messages": [{"role": "user", "content": "Explain 1+1 in one short sentence with thinking process."}],
            "max_tokens": 256,
            "temperature": 0.0
        }
        try:
            resp = requests.post(f"http://localhost:{port}/v1/chat/completions", json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                results[case] = {
                    "has_reasoning_content": "reasoning_content" in msg,
                    "content_length": len(msg.get("content", "")),
                    "reasoning_length": len(msg.get("reasoning_content", "")) if "reasoning_content" in msg else 0
                }
                print(f"Success: {results[case]}")
            else:
                results[case] = f"Error: {resp.text}"
        except Exception as e:
            results[case] = f"Request Failed: {e}"

        # cleanup
        proc.terminate()
        proc.wait()
    
    return results

if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Selection of smallest models (one from each family)
    test_targets = [
        ("Qwen-3.5-2B-Q4", 8101),
        ("Gemma-4-E2B-Q4", 8113),
        ("LFM-2.5-1.2B-Q4", 8121),
        ("GPT-OSS-20B-Q4", 8131),
        ("InternVL-3.5-2B-Q4", 8141),
        ("Next2-Air-Q4", 8142)
    ]

    summary = {}
    for name, port in test_targets:
        if name in config["models"]:
            cfg = config["models"][name]
            # Resolve path relative to host
            m_path = cfg["model_path"]
            mm_path = cfg.get("mmproj_path")
            summary[name] = test_model_reasoning(name, port, m_path, mm_path)
            
    print("\n\n=== FINAL REASONING TEST SUMMARY ===")
    print(json.dumps(summary, indent=2))
    with open("reasoning_test_report.json", "w") as f:
        json.dump(summary, f, indent=2)
