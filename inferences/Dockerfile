# Stage 1: Build the Go binary
FROM golang:1.25.5-bookworm AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- ONNX Runtime DEV (headers + libs สำหรับตอน compile) ---
RUN curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz \
    | tar -xzC /tmp \
    && mv /tmp/onnxruntime-linux-x64-1.23.2 /usr/local/onnxruntime

ENV CGO_ENABLED=1
ENV CGO_CFLAGS="-I/usr/local/onnxruntime/include"
ENV CGO_LDFLAGS="-L/usr/local/onnxruntime/lib -lonnxruntime"

COPY go.mod go.sum ./
RUN go mod download

COPY main.go .

RUN go build -o inference-app main.go


# Stage 2: Runtime image
FROM debian:bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# เruntime libs 
RUN curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz \
    | tar -xzC /tmp \
    && mv /tmp/onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so* /usr/lib/ \
    && ldconfig

COPY --from=builder /app/inference-app .

ENV LD_LIBRARY_PATH="/usr/lib"
ENV MODEL_PATH="/models/latest/cyclegan.onnx"
ENV PORT=8080

EXPOSE 8080
CMD ["./inference-app"]
