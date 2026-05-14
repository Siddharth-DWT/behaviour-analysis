# backend/Dockerfile — NEXUS monolithic backend
#
# Single image replaces 6 separate agent Dockerfiles:
#   services/voiceAgent/Dockerfile
#   services/language_agent/Dockerfile
#   services/conversation_agent/Dockerfile
#   services/video_agent/Dockerfile
#   services/fusion_agent/Dockerfile
#   services/api_gateway/Dockerfile
#
# Build context: ./backend
# Mounted at runtime (dev): ./shared → /app/shared, ./services → /app/services

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgles2 \
        libegl1 \
        ffmpeg \
        g++ \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch + torchaudio: CPU-only build (no CUDA in this container).
# Must be installed before requirements.txt so transformers picks up the right torch.
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Single worker — VideoAgentService uses its own ThreadPoolExecutor for CPU
# parallelism; multiple uvicorn workers would each spin up separate model pools
# and exhaust RAM. Scale horizontally via container replicas, not workers.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
