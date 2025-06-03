# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision \
    onnx onnxruntime flask pillow numpy requests

EXPOSE 8080

CMD ["python", "app.py"]