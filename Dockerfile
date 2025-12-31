FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api.py /app/
COPY search_orchestration.py /app/
COPY logging_config.py /app/

COPY agents/ /app/agents/
COPY llm/ /app/llm/

# Copy only the agent config
COPY configs/config_agents.yaml /app/configs/config_agents.yaml

# Copy model artifacts (FAISS, embeddings, mapping, metadata, memory)
COPY data/embeddings/ /app/data/embeddings/
COPY data/memory/ /app/data/memory/

# Expose API port
EXPOSE 8000

# Start FastAPI service
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

