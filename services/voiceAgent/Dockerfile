FROM python:3.11-slim

WORKDIR /app

# System deps for audio processing + git (needed for speechbrain develop branch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Layer 1: PyTorch CPU (cached unless torch version changes)
RUN pip install --no-cache-dir torch==2.4.1+cpu torchaudio==2.4.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Layer 2: Heavy ML deps (cached separately so network failures don't restart torch)
RUN pip install --no-cache-dir "matplotlib>=3.10.0" "filterpy>=1.4.5" "praat-parselmouth>=0.4.3"

# Layer 3: SpeechBrain from develop branch (large git clone, cached separately)
RUN pip install --no-cache-dir "speechbrain @ git+https://github.com/speechbrain/speechbrain.git@develop"

# Layer 4: Pyannote + its heavy telemetry deps (cached separately)
RUN pip install --no-cache-dir "pyannote.audio>=4.0"

# Layer 5: Remaining requirements
COPY requirements.txt .
RUN pip install --default-timeout=600 --retries 10 -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
