FROM python:3.11-slim

WORKDIR /app

# System dependencies required by sentence-transformers and lancedb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to avoid the 1.5GB CUDA version
# being pulled in by sentence-transformers
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY kenya-hiv-cdss/api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY kenya-hiv-cdss/app/ app/
COPY kenya-hiv-cdss/api/ api/

# Copy ambulance protocol PDFs
COPY kenya-hiv-cdss/Ambulensi*.pdf ./

# LanceDB index is built here on first startup and reused on subsequent starts
RUN mkdir -p lancedb

EXPOSE 8000

# OPENAI_API_KEY and OPENAI_BASE_URL must be passed at runtime via --env or docker-compose
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
