FROM python:3.11-slim

# OpenCV + MediaPipe system deps + ffmpeg for H.264 re-encoding
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

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download InsightFace buffalo_l model so the first analysis job
# does not stall on a runtime download.  Model lands in /root/.insightface/
# (outside the /app volume mount) and is baked into the image layer.
RUN python3 -c "\
import insightface; \
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=0, det_size=(640, 640)); \
print('buffalo_l pre-download complete')"

COPY . .

# Pre-download MediaPipe .task model files so the first analysis job
# does not stall on a runtime download.  Files land in /app/models/mediapipe/
# and are baked into the image layer alongside the source code.
RUN python3 -c "\
import os; os.makedirs('models/mediapipe', exist_ok=True); \
from feature_extractor import MediaPipeModelManager; \
mgr = MediaPipeModelManager('models/mediapipe'); \
mgr.get_face_landmarker_path(); \
mgr.get_pose_landmarker_path(); \
mgr.get_hand_landmarker_path(); \
mgr.get_gesture_recognizer_path(); \
mgr.get_face_detector_path(); \
print('MediaPipe models pre-download complete')"

EXPOSE 8004

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
