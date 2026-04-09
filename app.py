# bench_UI_simple/app.py
import gradio as gr
import os
import time
import pandas as pd
from model_manager import VLMModelManager

# Initialize the manager
manager = VLMModelManager()

def chat_interface(model_name, message, history, system_prompt, max_tokens, temperature, reasoning):
    user_text = message.get("text", "")
    user_files = message.get("files", [])
    
    # Process user input
    user_content = []
    if user_text:
        user_content.append({"type": "text", "text": user_text})
    for f in user_files:
        # Standard Gradio 6 image format
        if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
            user_content.append({"type": "image", "image": f})
        else:
            user_content.append({"type": "file", "file": f})
    
    # If only text exists, prefer a simple string for compatibility
    if len(user_content) == 1 and user_content[0]["type"] == "text":
        history.append({"role": "user", "content": user_text})
    else:
        history.append({"role": "user", "content": user_content})
    
    # Placeholder for streaming
    history.append({"role": "assistant", "content": ""})
    
    print(f"--- [DEBUG] History before stream: {history} ---")
    yield history, "Waiting...", "Waiting..."

    # Query model with stream
    image_path = user_files[0] if user_files else None
    
    for resp in manager.stream_query(model_name, user_text, image_path, system_prompt, max_tokens, temperature, reasoning=reasoning):
        if resp["status"] == "success":
            history[-1]["content"] = resp["text"]
            yield history, "Streaming...", "Streaming..."
        else:
            history[-1]["content"] = f"Error: {resp['message']}"
            yield history, "Error", "Error"
            return
    
    # Final update for TPS (since stream doesn't give precise TPS easily, we mark as Done)
    yield history, "Done (Stream)", "Done (Stream)"

def run_benchmark(model_names, test_text, system_prompt, max_tokens, temperature, reasoning):
    if not test_text: return pd.DataFrame(), "테스트 텍스트를 입력하세요."
    
    results = []
    total_models = len(model_names)
    
    for i, model_name in enumerate(model_names):
        yield pd.DataFrame(results), f"<div class='info-text'>진행 중: {model_name} ({i+1}/{total_models})</div>"
        
        input_tps_list = []
        output_tps_list = []
        durations = []
        
        # 20 Iterations
        for iter_idx in range(20):
            resp = manager.query(model_name, test_text, None, system_prompt, max_tokens, temperature, reasoning=reasoning)
            
            # Skip the first iteration (warmup)
            if iter_idx > 0 and resp["status"] == "success":
                input_tps_list.append(resp["input_tps"])
                output_tps_list.append(resp["output_tps"])
                durations.append(resp["duration"])
            
            time.sleep(0.1) # Cool down
            
        if input_tps_list:
            avg_input = sum(input_tps_list) / len(input_tps_list)
            avg_output = sum(output_tps_list) / len(output_tps_list)
            avg_duration = sum(durations) / len(durations)
            
            results.append({
                "Model": model_name,
                "Avg Input TPS (19-runs)": round(avg_input, 2),
                "Avg Output TPS (19-runs)": round(avg_output, 2),
                "Avg Latency (s)": round(avg_duration, 3)
            })
        else:
            results.append({
                "Model": model_name,
                "Avg Input TPS (19-runs)": "Failed",
                "Avg Output TPS (19-runs)": "Failed",
                "Avg Latency (s)": "Timeout / Error"
            })
            
    yield pd.DataFrame(results), "<div class='info-text'>벤치마크 완료</div>"

def get_status_table():
    status_dict = manager.get_running_status()
    rows = []
    for name, cfg in manager.models.items():
        m_status = status_dict.get(name, {"state": "Stopped", "reasoning": "-"})
        rows.append({
            "Model Name": name,
            "Port": cfg['port'],
            "Status": m_status["state"],
            "Reasoning": m_status["reasoning"],
            "Action": "Ready"
        })
    return pd.DataFrame(rows)

def toggle_server(model_name, action, reasoning=False):
    if action == "Start":
        manager.start_server(model_name, reasoning=reasoning)
    else:
        manager.stop_server(model_name)
    return get_status_table()

# UI Header
css = """
footer {visibility: hidden}
.status-box { background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
.info-text { font-size: 0.85em; color: #666; }
"""

with gr.Blocks(title="VLM Research Bench UI (Simple)") as demo:
    gr.Markdown("# 🚀 VLM Research Bench UI (Blackwell Optimized)")
    gr.Markdown("GPU: NVIDIA Blackwell GB10 | Arch: ARM64 | Environment: CUDA 13.0")
    
    # Pre-calculate initial reasoning state to avoid demo.load loop
    model_list = list(manager.models.keys())
    default_model = model_list[0] if model_list else None
    
    def get_initial_reasoning(model_name):
        if not model_name:
            return False, True, "<div class='info-text'>No models available</div>"
        info = manager.get_reasoning_info(model_name)
        status = info["status"]
        label = info["label"]
        if status == "forced_on":
            return True, False, f"<div class='info-text'>ℹ️ {label}</div>"
        elif status == "unsupported":
            return False, False, f"<div class='info-text'>⚠️ {label}</div>"
        else:
            return False, True, f"<div class='info-text'>✅ {label}</div>"

    init_val, init_inter, init_html = get_initial_reasoning(default_model)
    
    with gr.Tabs():
        # -- Tab 1: Playground --
        with gr.Tab("Playground (Qualitative)"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_list = list(manager.models.keys())
                    model_dropdown = gr.Dropdown(choices=model_list, value=model_list[0], label="모델 선택")
                    system_input = gr.Textbox(value="You are a helpful assistant.", label="System Prompt", lines=3)
                    with gr.Row():
                        max_tokens = gr.Slider(64, 4096, value=1024, step=64, label="Max Tokens")
                        temp = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature")
                    
                    reasoning_opt = gr.Checkbox(label="Reasoning (Chain of Thought)", value=init_val, interactive=init_inter)
                    reasoning_info = gr.HTML(init_html, container=False)
                
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="VLM Chat", height=600)
                    with gr.Row():
                        in_tps = gr.Textbox(label="Input Speed (TPS)", interactive=False)
                        out_tps = gr.Textbox(label="Output Speed (TPS)", interactive=False)
                    chat_input = gr.MultimodalTextbox(interactive=True, placeholder="Text or Image...", show_label=False)

            def update_reasoning_ui(model_name):
                if not model_name: 
                    return gr.update(), ""
                info = manager.get_reasoning_info(model_name)
                status = info["status"]
                label = info["label"]
                
                if status == "forced_on":
                    return gr.update(value=True, interactive=False), f"<div class='info-text'>ℹ️ {label}</div>"
                elif status == "unsupported":
                    return gr.update(value=False, interactive=False), f"<div class='info-text'>⚠️ {label}</div>"
                else:
                    return gr.update(interactive=True), f"<div class='info-text'>✅ {label}</div>"

            model_dropdown.change(
                update_reasoning_ui, 
                inputs=[model_dropdown], 
                outputs=[reasoning_opt, reasoning_info]
            )
            
            # Initial UI state (Moved to end)
            
            chat_input.submit(
                chat_interface,
                inputs=[model_dropdown, chat_input, chatbot, system_input, max_tokens, temp, reasoning_opt],
                outputs=[chatbot, in_tps, out_tps]
            )

        # -- Tab 2: Server Management --
        with gr.Tab("Server Management"):
            gr.Markdown("### 🛠 VLM 서버 개별 제어판")
            gr.Markdown("모델별 서버를 켜거나 꺼서 VRAM 자원을 관리할 수 있습니다.")
            
            with gr.Row():
                status_table = gr.Dataframe(
                    value=get_status_table(),
                    label="현재 서버 상태", 
                    interactive=False,
                    max_height=400
                )
            
            with gr.Row():
                with gr.Column():
                    manage_target = gr.Dropdown(choices=model_list, value=model_list[0], label="제어 대상 모델")
                    manage_reasoning = gr.Checkbox(label="Reasoning Mode (Start 시 반영)", value=init_val, interactive=init_inter)
                    manage_reasoning_info = gr.HTML(init_html, container=False)
                    with gr.Row():
                        start_btn = gr.Button("▶ Start Server", variant="primary")
                        stop_btn = gr.Button("⏹ Stop Server", variant="secondary")
                
                with gr.Column():
                    refresh_btn = gr.Button("🔄 상태 새로고침")

            # Event handlers
            start_btn.click(fn=lambda m, r: toggle_server(m, "Start", r), inputs=[manage_target, manage_reasoning], outputs=[status_table])
            stop_btn.click(fn=lambda m, r: toggle_server(m, "Stop", r), inputs=[manage_target, manage_reasoning], outputs=[status_table])
            refresh_btn.click(fn=get_status_table, outputs=[status_table])
            
            manage_target.change(
                update_reasoning_ui, 
                inputs=[manage_target], 
                outputs=[manage_reasoning, manage_reasoning_info]
            )
            
            # Manual Refresh only (Auto-timer removed for performance)

        # -- Tab 3: Benchmark --
        with gr.Tab("Benchmark (Quantitative)"):
            gr.Markdown("### 📊 정량적 성능 측정 (20회 반복, 첫 회 결과 제외)")
            with gr.Row():
                with gr.Column(scale=1):
                    bench_models = gr.CheckboxGroup(choices=model_list, value=model_list, label="측정 대상 모델")
                    bench_reasoning = gr.Checkbox(label="Enable Reasoning (COT) for Benchmark", value=False)
                    bench_text = gr.Textbox(value="Describe the image in detail.", label="Test Prompt")
                    bench_btn = gr.Button("🚀 벤치마크 시작", variant="primary")
                    bench_status = gr.HTML("<div class='info-text'>상태: 대기 중</div>", container=False)
                
                with gr.Column(scale=2):
                    bench_table = gr.Dataframe(label="결과 테이블", interactive=False, max_height=500)
            
            bench_btn.click(
                run_benchmark,
                inputs=[bench_models, bench_text, system_input, max_tokens, temp, bench_reasoning],
                outputs=[bench_table, bench_status]
            )

    # (init_ui and demo.load removed to prevent recursive UI sync loops in Gradio 6/Svelte 5)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, css=css)
