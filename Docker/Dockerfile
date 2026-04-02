# bench_UI_simple/Docker/Dockerfile
# Optimized for NVIDIA Blackwell (SM121) / ARM64 / CUDA 13.0

# --- Stage 1: Build llama-server ---
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y git cmake build-essential \
    libcurl4-openssl-dev python3 python3-pip

WORKDIR /app
RUN git clone https://github.com/ggerganov/llama.cpp.git

WORKDIR /app/llama.cpp
# Compile for Blackwell (sm_121) with explicit linker flags for stub libraries
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=121 \
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
    -DGGML_CUDA_F16=ON && \
    cmake --build build --config Release -j$(nproc) --target llama-server && \
    mkdir -p /app/libs && find build -name "*.so*" -exec cp -P {} /app/libs/ \;

# --- Stage 2: Runtime ---
FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y python3 python3-pip libcurl4 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy compiled binary and all necessary shared libraries
COPY --from=builder /app/llama.cpp/build/bin/llama-server /usr/local/bin/
RUN mkdir -p /usr/local/lib/llama.cpp
COPY --from=builder /app/libs/* /usr/local/lib/llama.cpp/
RUN echo "/usr/local/lib/llama.cpp" > /etc/ld.so.conf.d/llama_cpp.conf && ldconfig

# Install Python dependencies (Gradio, Requests, etc.)
RUN pip install --no-cache-dir --break-system-packages gradio requests pyyaml psutil pandas

# Set environment
ENV LLAMA_SERVER_PATH=/usr/local/bin/llama-server
ENV PYTHONUNBUFFERED=1

CMD ["python3", "app.py"]
