#!/bin/bash
# scripts/quantize_gemma4.sh
# Gemma 4 (26B-A4B & E2B) Multimodal Quantization & Upload Pipeline
# Optimized for NVIDIA Blackwell (DGX Spark)

set -e

# Configuration
REPO_DEST="sowilow/Gemma-4-DGX-Spark-GGUF"
MODELS=("google/gemma-4-26B-A4B-it" "google/gemma-4-E2B-it")
BASE_DIR="./models"
LLAMA_CPP_DIR="./llama.cpp"

mkdir -p $BASE_DIR

# 0. Ensure llama.cpp is ready (Builder stage logic)
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git $LLAMA_CPP_DIR
    cd $LLAMA_CPP_DIR && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j$(nproc) && cd ..
fi

for MODEL_ID in "${MODELS[@]}"; do
    MODEL_NAME=$(basename $MODEL_ID)
    echo "--- Processing $MODEL_NAME ---"

    # 1. Download HF Weights
    python3 scripts/hf_helper.py download $MODEL_ID "$BASE_DIR/$MODEL_NAME-hf"

    # 2. Convert to GGUF F16 (Main Model)
    python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$BASE_DIR/$MODEL_NAME-hf" \
        --outfile "$BASE_DIR/$MODEL_NAME-f16.gguf"

    # 3. Extract mmproj (Vision Tower)
    python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$BASE_DIR/$MODEL_NAME-hf" \
        --mmproj \
        --outfile "$BASE_DIR/gemma-4-vision-mmproj-f16.gguf"

    # 4. Quantize to Q4_K_M (Main Model)
    "$LLAMA_CPP_DIR/build/bin/llama-quantize" "$BASE_DIR/$MODEL_NAME-f16.gguf" \
        "$BASE_DIR/$MODEL_NAME-q4_k_m.gguf" Q4_K_M

    # 5. Upload to Hugging Face Hub
    python3 scripts/hf_helper.py upload $REPO_DEST "$BASE_DIR/$MODEL_NAME-q4_k_m.gguf" "$MODEL_NAME-q4_k_m.gguf"
    python3 scripts/hf_helper.py upload $REPO_DEST "$BASE_DIR/gemma-4-vision-mmproj-f16.gguf" "gemma-4-vision-mmproj-f16.gguf"

    echo "--- Finished $MODEL_NAME ---"
done

echo "Gemma 4 Multimodal Pipeline Complete."
