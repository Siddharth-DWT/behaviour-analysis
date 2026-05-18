FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download DistilBERT weights so the first language analysis job
# does not stall on a runtime HuggingFace Hub download.  Weights land
# in /root/.cache/huggingface/ and are baked into the image layer.
RUN python3 -c "\
from transformers import pipeline; \
pipeline('sentiment-analysis', \
    model='distilbert-base-uncased-finetuned-sst-2-english', \
    device=-1); \
print('DistilBERT pre-download complete')"

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
