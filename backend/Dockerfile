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

# ── MediaPipe models (5 files, ~240 MB, Google CDN, no auth) ─────────────────
# Downloaded to /app/models/mediapipe — matches MEDIAPIPE_MODEL_DIR at runtime.
# Without this, MediaPipeModelManager fetches each file on the first video session.
RUN python -c "\
import urllib.request, os; \
os.makedirs('models/mediapipe', exist_ok=True); \
models = {\
  'face_landmarker.task':        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',\
  'pose_landmarker_heavy.task':  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',\
  'hand_landmarker.task':        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',\
  'gesture_recognizer.task':     'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task',\
  'blaze_face_full_range.tflite':'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite',\
};\
[\
  (urllib.request.urlretrieve(url, f'models/mediapipe/{name}'), print(f'  {name} done'))\
  for name, url in models.items()\
];\
print('MediaPipe models ready.')"

# ── InsightFace buffalo_l (~400 MB, InsightFace CDN, no auth) ────────────────
# app.prepare() downloads weights and compiles ONNX graphs into
# /root/.insightface/models/buffalo_l/ for all future container runs.
RUN python -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=0, det_size=(320, 320)); \
print('InsightFace buffalo_l ready.')"

# ── Whisper / DistilBERT PyTorch / all-MiniLM-L6-v2: NOT baked ──────────────
# Whisper medium (~1.5 GB): skipped — TRANSCRIPTION_BACKEND=assemblyai means
#   AssemblyAI handles all transcription at runtime; local Whisper is never called.
#   faster-whisper downloads on first use if the backend is ever switched.
#
# DistilBERT PyTorch (~268 MB): skipped — ONNX INT8 model files already ship in
#   backend/models/distilbert-onnx-int8/ (volume-mounted at /app/models/).
#   The PyTorch fallback in language_agent/feature_extractor.py is only reached
#   when those ONNX files are absent, which never happens in practice.
#
# all-MiniLM-L6-v2 (~22 MB): skipped — only loaded when LLM_PROVIDER=anthropic.
#   With LLM_PROVIDER=openai the embedding path uses the OpenAI API; SentenceTransformer
#   is never instantiated.

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Single worker — VideoAgentService uses its own ThreadPoolExecutor for CPU
# parallelism; multiple uvicorn workers would each spin up separate model pools
# and exhaust RAM. Scale horizontally via container replicas, not workers.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
