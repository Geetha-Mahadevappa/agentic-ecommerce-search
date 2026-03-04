# Use CUDA base image for GPU support
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Install GPU PyTorch first to match CUDA version
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122

# Install the rest of the dependencies (faiss-gpu included)
RUN pip install --no-cache-dir -r requirements.txt

# Ensure config and data directories exist
RUN mkdir -p /app/configs /app/data/embeddings /app/data/memory

# Copy app code
COPY api.py /app/
COPY search_orchestration.py /app/
COPY logging_config.py /app/
COPY agents/ /app/agents/
COPY llm/ /app/llm/
COPY configs/config_agents.yaml /app/configs/
COPY data/embeddings/ /app/data/embeddings/
COPY data/memory/ /app/data/memory/

# Expose API port
EXPOSE 8000

# Run API with longer timeout for model loading
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120"]
