FROM python:3.10-slim

# Environment settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4

WORKDIR /app

# System dependencies
# - openblas/omp for sentence-transformers
# - curl for healthchecks
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure config directory exists before copying
RUN mkdir -p /app/configs

# Copy application code
COPY api.py /app/
COPY search_orchestration.py /app/
COPY logging_config.py /app/

COPY agents/ /app/agents/
COPY llm/ /app/llm/

# Copy only the agent config
RUN mkdir -p /app/configs
COPY configs/config_agents.yaml /app/configs/config_agents.yaml

# Copy artifacts
RUN mkdir -p /app/data/embeddings
COPY data/embeddings/ /app/data/embeddings/

RUN mkdir -p /app/data/memory
COPY data/memory/ /app/data/memory/

EXPOSE 8000

# Uvicorn timeout increased because model loading is slow on first run
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120"]
