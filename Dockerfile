# Face Authentication Platform — API service container.
# All base images and packages below are open source (spec Section 6,
# Section 18 Phase 1: "container-based; deployable on AWS, GCP, or Azure").
FROM python:3.12-slim

# OpenCV / mediapipe runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV FACE_AUTH_DATA_DIR=/data
VOLUME ["/data"]

EXPOSE 8000

# ArcFace weights, if you have them, get mounted at this path (see
# README "Swapping in ArcFace") -- app/ml/embedder.py checks for them
# automatically and activates the production embedder when present.
ENV ARCFACE_MODEL_PATH=/models/buffalo_l/w600k_r50.onnx

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
