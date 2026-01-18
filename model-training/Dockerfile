FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    onnx \
    tqdm \
    pillow \
    numpy

COPY model_training.py .

RUN mkdir -p data/trainA data/trainB saved_models

ENV MODE=train

CMD ["python", "model_training.py"]