import torch
import json
import os
import sys
from safetensors.torch import load_file
import gguf

def extract_mtp_draft(model_dir, output_path):
    print(f"Loading MTP tensors from {model_dir}...")
    
    # 35B A3B original has MTP in shards 13 and 14
    mtp_tensors = {}
    for i in range(13, 15):
        shard_path = os.path.join(model_dir, f"model.safetensors-{i:05d}-of-00014.safetensors")
        if os.path.exists(shard_path):
            shard = load_file(shard_path)
            for k, v in shard.items():
                if k.startswith("mtp."):
                    mtp_tensors[k] = v
    
    if not mtp_tensors:
        print("No MTP tensors found.")
        return

    print(f"Extracted {len(mtp_tensors)} MTP tensors. Creating Draft GGUF...")

    with open(os.path.join(model_dir, "config.json"), "r") as f:
        full_config = json.load(f)
    config = full_config.get("text_config", full_config)

    # Use Qwen2 arch for the draft (as it's just a transformer block)
    writer = gguf.GGUFWriter(output_path, "qwen2")
    
    writer.add_architecture()
    writer.add_name("Qwen3.5-35B-MTP-Draft")
    writer.add_context_length(config.get("max_position_embeddings", 32768))
    writer.add_embedding_length(config["hidden_size"])
    writer.add_block_count(1)
    
    # Qwen 3.5 35B intermediate_size for Experts is often different. 
    # Use hidden_size * 3.5 or similar, but for draft we just need the tensor shape to match.
    writer.add_feed_forward_length(2048) # Filler, tensors will define shape
    writer.add_head_count(config["num_attention_heads"])
    writer.add_head_count_kv(config["num_key_value_heads"])
    writer.add_layer_norm_rms_eps(config.get("rms_norm_eps", 1e-6))
    writer.add_rope_freq_base(config.get("rope_theta", 10000.0))
    
    # Add basic tokenizer metadata (required for llama.cpp to load it as a model)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen35")

    tensor_map = {
        "mtp.layers.0.self_attn.q_proj.weight": "blk.0.attn_q.weight",
        "mtp.layers.0.self_attn.k_proj.weight": "blk.0.attn_k.weight",
        "mtp.layers.0.self_attn.v_proj.weight": "blk.0.attn_v.weight",
        "mtp.layers.0.self_attn.o_proj.weight": "blk.0.attn_output.weight",
        "mtp.layers.0.mlp.gate_proj.weight": "blk.0.ffn_gate.weight",
        "mtp.layers.0.mlp.up_proj.weight": "blk.0.ffn_up.weight",
        "mtp.layers.0.mlp.down_proj.weight": "blk.0.ffn_down.weight",
        "mtp.layers.0.input_layernorm.weight": "blk.0.attn_norm.weight",
        "mtp.layers.0.post_attention_layernorm.weight": "blk.0.ffn_norm.weight",
    }
    
    for old_k, new_k in tensor_map.items():
        if old_k in mtp_tensors:
            writer.add_tensor(new_k, mtp_tensors[old_k].float().numpy())
    
    if "mtp.fc.weight" in mtp_tensors:
         # Map to enorm/eh_proj logic or similar if needed, 
         # but for self-speculative draft, we focus on the blocks.
         writer.add_tensor("mtp_fc.weight", mtp_tensors["mtp.fc.weight"].float().numpy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    print(f"Successfully created MTP Draft GGUF: {output_path}")

if __name__ == "__main__":
    extract_mtp_draft("/home/laon/Desktop/Qwen3.5/Qwen3.5-35B-A3B", "/home/laon/Desktop/bench_UI/bench_UI_simple/models/qwen3.5-35b-mtp-draft.gguf")
