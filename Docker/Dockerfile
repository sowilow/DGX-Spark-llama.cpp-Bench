# bench_UI_simple/Docker/Dockerfile
# Uses pre-built llama.cpp binaries from host (no compilation inside Docker)
# Optimized for NVIDIA Blackwell (SM121) / ARM64 / CUDA 13.0

FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y python3 python3-pip libcurl4 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Copy pre-built llama-server and llama-quantize from host build
COPY llama.cpp/build/bin/llama-server /usr/local/bin/
COPY llama.cpp/build/bin/llama-quantize /usr/local/bin/

# Copy ALL required shared libraries (including libmtmd.so for Gemma-4)
COPY llama.cpp/build/bin/libllama.so* /usr/local/lib/
COPY llama.cpp/build/bin/libmtmd.so* /usr/local/lib/
COPY llama.cpp/build/bin/libggml*.so* /usr/local/lib/

# Copy conversion scripts for self-contained operation
RUN mkdir -p /opt/llama.cpp
COPY llama.cpp/convert_hf_to_gguf.py /opt/llama.cpp/
COPY llama.cpp/gguf-py /opt/llama.cpp/gguf-py

# Refresh library cache
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
RUN ldconfig

# Install Python dependencies
RUN pip install --no-cache-dir --break-system-packages \
    "gradio>=5.0.0" requests pyyaml psutil pandas huggingface_hub \
    sentencepiece gguf numpy tqdm

# Set environment
ENV LLAMA_SERVER_PATH=/usr/local/bin/llama-server
ENV LLAMA_CONVERT_SCRIPT=/opt/llama.cpp/convert_hf_to_gguf.py
ENV PYTHONUNBUFFERED=1

CMD ["python3", "app.py"]
