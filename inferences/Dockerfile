# Stage 1: Build the Go binary
FROM golang:1.25.5-bookworm AS builder

WORKDIR /app

# Copy Go module files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY main.go .

# Build the binary
RUN go build -o inference-app main.go

# Stage 2: Create the runtime image
FROM debian:bookworm-slim

WORKDIR /app

# Install dependencies (certificates for downloading, etc.)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Download ONNX Runtime (Linux x64)
RUN curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz \
    | tar -xzC /tmp \
    && mv /tmp/onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so* /usr/lib/ \
    && ldconfig

# Copy the binary from the builder stage
COPY --from=builder /app/inference-app .

# Create directories for data (to be mounted)
RUN mkdir -p /data/model /data/input /data/output

# Set environment variable for the app (Optional override)
ENV LD_LIBRARY_PATH="/usr/lib"

# Command to run
CMD ["./inference-app"]