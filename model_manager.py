# bench_UI_simple/model_manager.py
import subprocess
import time
import os
import sys
import requests
import psutil
import signal
import yaml
import json
from huggingface_hub import hf_hub_download

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
        self.running_reasoning = {} # {model_name: "on" or "off"}
        self.server_bin = "/usr/local/bin/llama-server"
        
        # Reasoning Configuration (Problem 2, 3 Mitigation)
        self.reasoning_config = {
            "GPT-OSS": {"status": "forced_on", "label": "Always-on Reasoning (Model Design)"},
            "LFM": {"status": "unsupported", "label": "Reasoning not supported for this model"},
            "InternVL": {"status": "unsupported", "label": "Reasoning not supported for this model"},
            "Qwen": {"status": "supported", "label": "Toggleable Reasoning"},
            "Gemma": {"status": "supported", "label": "Toggleable Reasoning"},
            "Next2": {"status": "supported", "label": "Toggleable Reasoning"}
        }
        
        print("--- Environment Check (DGX Spark / Arm / CUDA 13) ---")
        self._detect_hardware()
        self._check_and_download_models()
        
        # 가동 시 모든 모델을 미리 로드할지 결정 (config.yaml 기반)
        if self.hw_config.get('pre_start_all', True):
            self.start_all_servers()

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

    def _check_and_download_models(self):
        missing_files = []
        models_dir = os.path.join(self.base_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        for model_name, cfg in self.models.items():
            for key in ['model_path', 'mmproj_path']:
                path = cfg.get(key)
                if path:
                    abs_path = os.path.join(self.base_dir, path)
                    # 파일이 없거나 크기가 너무 작으면(1MB 이하) 누락된 것으로 간주
                    if not os.path.exists(abs_path) or os.path.getsize(abs_path) < 1024 * 1024:
                        missing_files.append({
                            'repo_id': cfg.get('repo_id'),
                            'filename': os.path.basename(path),
                            'target_path': abs_path,
                            'model_name': model_name
                        })

        if not missing_files:
            print("모든 모델 파일이 존재합니다.")
            return

        print(f"\n[!] 다음 {len(missing_files)}개의 모델 파일이 누락되었습니다:")
        for f in missing_files:
            print(f" - {f['filename']} (from {f['repo_id']})")
        
        # Check for environment variable or TTY
        auto_download = os.environ.get("AUTO_DOWNLOAD", "").strip().lower() == "y"
        is_tty = sys.stdin.isatty()

        if auto_download:
            print("\n환경 변수(AUTO_DOWNLOAD=Y)가 설정되어 자동 다운로드를 시작합니다.")
        elif not is_tty:
            print("\n[!] 비대화형 환경(Docker 등)에서 실행 중이며 승인을 위한 터미널 입력을 받을 수 없습니다.")
            print("모델 다운로드를 자동화하려면 환경 변수 'AUTO_DOWNLOAD=Y'를 설정해 주세요.")
            return
        else:
            confirm = input("\n모델을 지금 다운로드하시겠습니까? (Y/n): ").strip().lower()
            if confirm != 'y':
                print("다운로드를 취소했습니다. 서버가 정상적으로 작동하지 않을 수 있습니다.")
                return

        print("\n--- 순차 다운로드 시작 ---")
        for i, f in enumerate(missing_files):
            print(f"[{i+1}/{len(missing_files)}] 다운로드 중: {f['filename']}...")
            try:
                # Special case: Qwen3.5_35B shares 2B mmproj, so it might be in either repo
                # But we'll try the assigned repo_id first
                hf_hub_download(
                    repo_id=f['repo_id'],
                    filename=f['filename'],
                    local_dir=models_dir,
                    local_dir_use_symlinks=False
                )
                print(f"완료: {f['filename']}")
            except Exception as e:
                print(f"오류 발생 ({f['filename']}): {e}")
        print("--- 모든 다운로드 작업 완료 ---\n")

    def start_all_servers(self):
        print("\n--- 모든 모델 서버 일괄 가동 시작 (Pre-start) ---")
        for model_name in self.models.keys():
            success = self.start_server(model_name)
            if not success:
                print(f"[!] {model_name} 서버 가동에 실패했습니다.")
        print("--- 모든 모델 서버 가동 작업 완료 ---\n")

    def start_server(self, model_name, reasoning=None):
        # If already running, return
        if model_name in self.processes and self.processes[model_name].poll() is None:
            return True
            
        if reasoning is None:
            reasoning = self.hw_config.get('reasoning', 'off')

        # reasoning이 bool일 경우 문자열 "on"/"off"로 변환
        if isinstance(reasoning, bool):
            reasoning = "on" if reasoning else "off"

        # Apply Model-specific overrides
        family = next((f for f in self.reasoning_config if f in model_name), None)
        if family:
            status = self.reasoning_config[family]["status"]
            if status == "forced_on": reasoning = "on"
            elif status == "unsupported": reasoning = "off"

        cfg = self.models[model_name]
        port = cfg['port']
        model_path = self._resolve_path(cfg['model_path'])
        
        # Build command
        cmd = [
            self.server_bin,
            "--host", "0.0.0.0",
            "-m", model_path,
            "--port", str(port),
            "-ngl", str(self.hw_config['gpu_layers']),
            "--ctx-size", str(self.hw_config['ctx_size']),
            "--flash-attn", self.hw_config['flash_attn'],
            "--reasoning", reasoning
        ]

        # Add mmproj if available (for VL models)
        mmproj_rel_path = cfg.get('mmproj_path')
        if mmproj_rel_path:
            mmproj_path = self._resolve_path(mmproj_rel_path)
            cmd.extend(["--mmproj", mmproj_path])
        
        print(f"--- [DEBUG] Starting server for {model_name} on port {port} (Reasoning: {reasoning}) ---")
        log_path = os.path.join(self.base_dir, f"server_{model_name}.log")
        log_file = open(log_path, "w", buffering=1)
        
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, preexec_fn=os.setsid)
        self.processes[model_name] = proc
        self.running_reasoning[model_name] = reasoning
        
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

    def stop_server(self, model_name):
        if model_name in self.processes:
            proc = self.processes[model_name]
            if proc.poll() is None:
                print(f"--- [DEBUG] Stopping server for {model_name} ---")
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    proc.wait(timeout=5)
                except Exception as e:
                    print(f"Error stopping {model_name}: {e}")
            del self.processes[model_name]
            return True
        return False

    def get_running_status(self):
        status = {}
        for name in self.models.keys():
            if name in self.processes and self.processes[name].poll() is None:
                status[name] = {
                    "state": "Running",
                    "reasoning": self.running_reasoning.get(name, "off")
                }
            else:
                status[name] = {
                    "state": "Stopped",
                    "reasoning": "-"
                }
        return status

    def query(self, model_name, prompt, image_path=None, system_prompt="You are a helpful assistant.", max_tokens=512, temperature=0.7, reasoning=None):
        if reasoning is None:
            reasoning = self.hw_config.get('reasoning', 'off')
        
        # reasoning이 bool일 경우 문자열 "on"/"off"로 변환
        if isinstance(reasoning, bool):
            reasoning = "on" if reasoning else "off"
            
        # 현재 실행 중인 서버의 reasoning 모드와 다를 경우 재시작
        current_reasoning = self.running_reasoning.get(model_name)
        if current_reasoning and current_reasoning != reasoning:
            print(f"--- [DEBUG] Reasoning mode change detected for {model_name} ({current_reasoning} -> {reasoning}). Restarting... ---")
            self.stop_server(model_name)

        # Ensure server is running
        if not self.start_server(model_name, reasoning=reasoning):
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
            response = requests.post(url, json=payload, timeout=300)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                msg = result["choices"][0]["message"]
                text = msg.get("content", "")
                reasoning_text = msg.get("reasoning_content", "")
                
                # Prepend reasoning if available
                if reasoning_text:
                    text = f"<details><summary>Thought</summary>\n\n{reasoning_text}\n\n</details>\n\n{text}"
                
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

    def stream_query(self, model_name, prompt, image_path=None, system_prompt="You are a helpful assistant.", max_tokens=512, temperature=0.7, reasoning=None):
        if reasoning is None:
            reasoning = self.hw_config.get('reasoning', 'off')
        
        if isinstance(reasoning, bool):
            reasoning = "on" if reasoning else "off"
            
        # Current reasoning mode check and restart if needed
        current_reasoning = self.running_reasoning.get(model_name)
        if current_reasoning and current_reasoning != reasoning:
            self.stop_server(model_name)

        if not self.start_server(model_name, reasoning=reasoning):
            yield {"status": "error", "message": f"Failed to start server for {model_name}"}
            return
            
        port = self.models[model_name]['port']
        url = f"http://localhost:{port}/v1/chat/completions"
        
        # Build messages (Simplified for stream)
        messages = [{"role": "system", "content": system_prompt}]
        content = []
        if image_path:
            import base64
            with open(image_path, "rb") as f:
                img_str = base64.b64encode(f.read()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}})
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
        
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }
        
        full_content = ""
        full_reasoning = ""
        
        try:
            response = requests.post(url, json=payload, stream=True, timeout=300)
            if response.status_code != 200:
                yield {"status": "error", "message": response.text}
                return

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        if decoded_line == "data: [DONE]": break
                        try:
                            chunk = json.loads(decoded_line[6:])
                            delta = chunk["choices"][0]["delta"]
                            
                            content_piece = delta.get("content", "")
                            reasoning_piece = delta.get("reasoning_content", "")
                            
                            if reasoning_piece:
                                full_reasoning += reasoning_piece
                            if content_piece:
                                full_content += content_piece
                            
                            # Build display text
                            display_text = ""
                            if full_reasoning:
                                display_text = f"<details open><summary>Thinking...</summary>\n\n{full_reasoning}\n\n</details>\n\n"
                            display_text += full_content
                            
                            yield {"status": "success", "text": display_text, "input_tps": 0, "output_tps": 0}
                        except Exception as e:
                            print(f"--- [DEBUG] Error parsing stream chunk: {e} ---")
                            pass
        except Exception as e:
            print(f"--- [DEBUG] Stream query exception: {e} ---")
            yield {"status": "error", "message": str(e)}

    def get_reasoning_info(self, model_name):
        family = next((f for f in self.reasoning_config if f in model_name), None)
        if family:
            return self.reasoning_config[family]
        return {"status": "supported", "label": "Toggleable Reasoning"}

    def stop_all(self):
        for name, proc in self.processes.items():
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except:
                pass
        self.processes = {}
