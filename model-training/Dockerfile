# model-training/Dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

RUN pip install mlflow pillow tqdm onnx

COPY . /app

CMD ["python", "model_training.py"]