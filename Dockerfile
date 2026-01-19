# Stage 1: Build the Go binary
FROM golang:1.25.5-bookworm AS builder

WORKDIR /app

# Copy Go module files from the 'inferences' directory
COPY inferences/go.mod inferences/go.sum ./
RUN go mod download

# Copy source code from the 'inferences' directory
COPY inferences/main.go .

# Build the binary
# CGO_ENABLED=1 is required for the ONNX Runtime Go wrapper
RUN CGO_ENABLED=1 GOOS=linux go build -o inference-app main.go

# Stage 2: Create the runtime image
FROM debian:bookworm-slim

WORKDIR /app

# Install dependencies (certificates, curl, libgomp1 for OpenMP support)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Download ONNX Runtime (Linux x64) v1.23.2
# We use the /usr/lib path so the system linker finds it automatically
RUN curl -L https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz \
    | tar -xzC /tmp \
    && mv /tmp/onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so* /usr/lib/ \
    && ldconfig

# Copy the binary from the builder stage
COPY --from=builder /app/inference-app .

# Copy the ONNX Model into the container
# This makes the container self-sufficient for Render deployment
COPY model-path/cyclegan_horse2zebra.onnx ./model.onnx

# Set Environment Variables
ENV LD_LIBRARY_PATH="/usr/lib"
ENV MODEL_PATH="./model.onnx"
ENV PORT=8080

# Expose the port for Render
EXPOSE 8080

# Command to run the server
CMD ["./inference-app"]