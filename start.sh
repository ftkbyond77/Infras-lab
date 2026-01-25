#!/bin/bash
set -e

# ==============================================================================
# 0. LOGGING & CONFIGURATION
# ==============================================================================
LOG_FILE="./startup.log"
rm -f "$LOG_FILE"
touch "$LOG_FILE"

# Redirect stdout and stderr to both console and log file
exec > >(tee -a "$LOG_FILE") 2>&1

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN} Starting MLOps Lifecycle Management Script${NC}"

# ==============================================================================
# 1. CLEANUP & ENVIRONMENT SETUP
# ==============================================================================
cleanup() {
    echo -e "\n${RED} Shutting down MLOps Platform...${NC}"
    docker compose down
    echo -e "${GREEN} Shutdown Complete${NC}"
    exit
}
trap cleanup SIGINT SIGTERM

# Dynamically set the absolute path for the shared volume
# This is crucial so Airflow (Container) sees the exact same files as Go (Container)
export HOST_MODEL_STORE_PATH="$(pwd)/model_store"
echo -e "${BLUE} Host Model Store Path detected: $HOST_MODEL_STORE_PATH${NC}"

# Update .env file dynamically so Docker Compose picks up the path
if grep -q "HOST_MODEL_STORE_PATH=" .env; then
    sed -i "s|HOST_MODEL_STORE_PATH=.*|HOST_MODEL_STORE_PATH=$HOST_MODEL_STORE_PATH|" .env
else
    echo "HOST_MODEL_STORE_PATH=$HOST_MODEL_STORE_PATH" >> .env
fi

# ==============================================================================
# 2. LIFECYCLE BOOTSTRAPPING (Filesystem Layer)
# ==============================================================================
echo -e "${BLUE} Bootstrapping Project Directory Structure...${NC}"

# 2.1 Data Pipeline Preparation (For Data Ingestion DAG)
# Creates the folders where the Data DAG will look for images
mkdir -p model_store/data/raw/trainA
mkdir -p model_store/data/raw/trainB
mkdir -p model_store/data/processed/trainA
mkdir -p model_store/data/processed/trainB

# 2.2 Model Lifecycle Preparation (For Training & Deployment DAGs)
# Creates folders for 'Candidate' models (post-training) and 'Latest' (production)
mkdir -p model_store/candidates
mkdir -p model_store/latest

# 2.3 Data Seeding (Simulate Data Ingestion)
# Checks if we have local data and moves it to the 'raw' folder so the pipeline has input
if [ -z "$(ls -A model_store/data/raw/trainA)" ]; then
    echo "    Seeding raw data from local 'model-training/data'..."
    if [ -d "model-training/data/trainA" ]; then
        cp -r model-training/data/trainA/* model_store/data/raw/trainA/ 2>/dev/null || true
        cp -r model-training/data/trainB/* model_store/data/raw/trainB/ 2>/dev/null || true
    else
        echo "    Warning: Local data not found. Please put images in model_store/data/raw/ manually."
    fi
fi

# 2.4 Model Seeding (Ensure Inference Service works on startup)
SOURCE_MODEL="model-path/cyclegan_horse2zebra.onnx"
DEST_MODEL="model_store/latest/cyclegan.onnx"

if [ ! -f "$DEST_MODEL" ]; then
    if [ -f "$SOURCE_MODEL" ]; then
        echo "    Seeding initial Production Model..."
        cp "$SOURCE_MODEL" "$DEST_MODEL"
    else
        echo -e "${RED}    Critical: Base model not found! Inference API will fail until first training run.${NC}"
    fi
else
    echo "    Production model validation: OK."
fi

# Grant permissions so Docker containers (Airflow user) can read/write
chmod -R 777 model_store
chmod -R 777 airflow/dags

# ==============================================================================
# 3. BUILD PHASE
# ==============================================================================
echo -e "${BLUE} Building Docker Images (No Cache)...${NC}"
# --no-cache ensures we pick up the latest code changes in Python and Go
docker compose build --no-cache

# ==============================================================================
# 4. SERVICE ORCHESTRATION (Startup Sequence)
# ==============================================================================
echo -e "${BLUE} Initializing Database Layer...${NC}"
docker compose up -d postgres

echo "    Waiting for Postgres health check..."
until docker compose exec -T postgres pg_isready -U airflow; do
    echo "      ...waiting for DB"
    sleep 2
done

echo -e "${BLUE} Initializing Airflow Core...${NC}"
docker compose up -d airflow-api-server

echo "    Waiting for Airflow API..."
sleep 8

# ==============================================================================
# 5. MLOPS CONFIGURATION (Airflow Setup)
# ==============================================================================
echo -e "${BLUE} Configuring Airflow Metadata & Connections...${NC}"

echo "    Running Database Migrations..."
docker compose exec -T airflow-api-server airflow db migrate > /dev/null 2>&1

echo "    Creating Admin User..."
docker compose exec -T airflow-api-server airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com > /dev/null 2>&1 || true

echo "    Connecting Airflow to Inference Service..."
# This allows DAG 05 (Deployment) to trigger the 'Hot Reload' on the Go service
docker compose exec -T airflow-api-server airflow connections delete inference_server > /dev/null 2>&1 || true
docker compose exec -T airflow-api-server airflow connections add 'inference_server' \
    --conn-type 'http' \
    --conn-host 'inference' \
    --conn-port '8080'

# ==============================================================================
# 6. FINAL LAUNCH
# ==============================================================================
echo -e "${GREEN} Launching Full Stack (Scheduler, Inference, MLflow, Frontend)...${NC}"
docker compose up -d

echo "    Waiting for services to stabilize..."
sleep 5

# Function to open browser based on OS
open_url() {
    if [[ "$OSTYPE" == "darwin"* ]]; then open "$1"; elif [[ "$OSTYPE" == "linux-gnu"* ]]; then xdg-open "$1"; fi
}

open_url "http://localhost:8080" # Apache Airflow Service
open_url "http://localhost:5000" # MLflow Service
open_url "http://localhost:3000" # Frontend Service

echo -e "\n${GREEN}================================================================${NC}"
echo -e "    MLOPS LIFECYCLE MANAGER - STATUS: ONLINE"
echo -e "================================================================"
echo -e "    1. Orchestrator:  http://localhost:8080  (User: admin / Pass: admin)"
echo -e "    2. Experiment:    http://localhost:5000  (MLflow)"
echo -e "    3. Inference API: http://localhost:8081/health"
echo -e "    4. Frontend:      http://localhost:3000"
echo -e "================================================================"
echo -e "    Model Artifacts:  $HOST_MODEL_STORE_PATH"
echo -e "${BLUE}    NEXT STEP: Go to Airflow and unpause '01_data_pipeline' to start.${NC}"
echo -e "================================================================"

# Tail logs for the most critical services: Scheduler (Pipeline) and Inference (Deployment)
docker compose logs -f airflow-scheduler inference