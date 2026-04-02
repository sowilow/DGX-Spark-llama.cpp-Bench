# bench_UI_simple/model_manager.py
import subprocess
import time
import os
import requests
import psutil
import signal
import yaml

class VLMModelManager:
    def __init__(self):
        # Resolve path relative to this script
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(self.base_dir, "config/config.yaml")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = self.config['models']
        self.hw_config = self.config['hardware']
        self.processes = {} # {model_name: process}
        self.server_bin = "/usr/local/bin/llama-server"
        
        print("--- Environment Check (DGX Spark / Arm / CUDA 13) ---")
        self._detect_hardware()

    def _detect_hardware(self):
        try:
            gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"]).decode().strip()
            print(f"GPU Detected: {gpu_info}")
        except:
            print("Warning: nvidia-smi not found.")

    def _resolve_path(self, p):
        abs_p = os.path.abspath(os.path.join(self.base_dir, p))
        real_p = os.path.realpath(abs_p)
        if not os.path.exists(real_p):
            # Emergency search in common container mount points
            basename = os.path.basename(p)
            for search_root in ["/app/models", "./models"]:
                full_search_path = os.path.join(self.base_dir, search_root) if search_root.startswith(".") else search_root
                if os.path.exists(full_search_path):
                    for root, dirs, files in os.walk(full_search_path):
                        if basename in files:
                            return os.path.join(root, basename)
        return real_p

    def start_server(self, model_name):
        # If already running, return
        if model_name in self.processes and self.processes[model_name].poll() is None:
            return True
            
        cfg = self.models[model_name]
        port = cfg['port']
        model_path = self._resolve_path(cfg['model_path'])
        mmproj_path = self._resolve_path(cfg['mmproj_path'])
        
        # Build command
        cmd = [
            self.server_bin,
            "-m", model_path,
            "--mmproj", mmproj_path,
            "--port", str(port),
            "-ngl", str(self.hw_config['gpu_layers']),
            "--ctx-size", str(self.hw_config['ctx_size']),
            "--flash-attn", self.hw_config['flash_attn'],
            "--reasoning", self.hw_config['reasoning']
        ]
        
        print(f"--- [DEBUG] Starting server for {model_name} on port {port} ---")
        log_path = os.path.join(self.base_dir, f"server_{model_name}.log")
        log_file = open(log_path, "w", buffering=1)
        
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, preexec_fn=os.setsid)
        self.processes[model_name] = proc
        
        # Wait for ready
        max_retries = 60
        for i in range(max_retries):
            try:
                resp = requests.get(f"http://localhost:{port}/health", timeout=1)
                if resp.status_code == 200:
                    print(f"Server {model_name} (Port {port}) is ready.")
                    return True
            except:
                pass
            time.sleep(1)
        return False

    def query(self, model_name, prompt, image_path=None, system_prompt="You are a helpful assistant.", max_tokens=512, temperature=0.7):
        # Ensure server is running
        if not self.start_server(model_name):
            return {"status": "error", "message": f"Failed to start server for {model_name}"}
            
        port = self.models[model_name]['port']
        url = f"http://localhost:{port}/v1/chat/completions"
        
        messages = [{"role": "system", "content": system_prompt}]
        content = []
        if image_path:
            import base64
            with open(image_path, "rb") as f:
                img_str = base64.b64encode(f.read()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
            })
            if "<image>" not in prompt: prompt = f"<image>\n{prompt}"
        
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
        
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        start_time = time.time()
        try:
            response = requests.post(url, json=payload, timeout=120)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                text = result["choices"][0]["message"]["content"]
                usage = result.get("usage", {})
                timings = result.get("timings", {})
                
                return {
                    "text": text,
                    "input_tps": timings.get("prompt_per_second", 0),
                    "output_tps": timings.get("predicted_per_second", 0),
                    "duration": end_time - start_time,
                    "status": "success"
                }
            return {"status": "error", "message": response.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def stop_all(self):
        for name, proc in self.processes.items():
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except:
                pass
        self.processes = {}
