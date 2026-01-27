# model-training/Dockerfile
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow \
    torch torchvision torchaudio \
    pillow tqdm numpy

COPY model_training.py .

ENV MLFLOW_URI=http://mlflow:5000

CMD ["python", "model_training.py"]