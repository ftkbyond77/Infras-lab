#!/bin/bash
set -e


LOG_FILE="./startup.log"

rm -f "$LOG_FILE"
touch "$LOG_FILE"

exec > >(tee -a "$LOG_FILE") 2>&1

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

cleanup() {
    echo -e "\n${RED} Shutting down...${NC}"
    docker compose down
    echo -e "${GREEN} Shutdown Complete${NC}"
    exit
}

trap cleanup SIGINT SIGTERM

echo -e "${GREEN} MLOps Factory Reset & Start${NC}"

echo -e "${GREEN} Building Images (No Cache)...${NC}"
docker compose build --no-cache

if [ ! -f .env ]; then
    echo -e "${RED} Error: .env file not found${NC}"
    exit 1
fi

echo " Setting up shared volumes..."
mkdir -p model_store/latest
mkdir -p model_store/candidates

SOURCE_MODEL="model-path/cyclegan_horse2zebra.onnx"
DEST_MODEL="model_store/latest/cyclegan.onnx"

if [ ! -f "$DEST_MODEL" ]; then
    echo " Warning: No model found in production volume."
    if [ -f "$SOURCE_MODEL" ]; then
        echo " Bootstrapping production model from: $SOURCE_MODEL"
        cp "$SOURCE_MODEL" "$DEST_MODEL"
    else
        echo -e "${RED} Critical: Source model '$SOURCE_MODEL' not found! Inference will fail.${NC}"
    fi
else
    echo " Production model exists."
fi

chmod -R 777 model_store
chmod -R 777 airflow/dags

echo " Starting Database Service..."
docker compose up -d postgres

echo " Waiting for Database to be healthy..."
until docker compose exec -T postgres pg_isready -U airflow; do
    echo "   ...waiting for Postgres"
    sleep 2
done

echo " Starting Airflow API container..."
docker compose up -d airflow-api-server

echo " Waiting for Airflow API container..."
sleep 5

echo " Running Airflow Migrations..."
docker compose exec airflow-api-server airflow db migrate

echo " Creating Admin User (if not exists)..."
docker compose exec airflow-api-server airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com || true

echo -e "${GREEN} Launching All Services...${NC}"
docker compose up -d

echo " Opening Interfaces..."
sleep 5

open_url() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "$1"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open "$1"
    elif [[ "$OSTYPE" == "msys" ]]; then
        start "$1"
    fi
}

open_url "http://localhost:3000"
open_url "http://localhost:8080"

echo -e "\n${GREEN}===================================================${NC}"
echo -e "   Frontend:   http://localhost:3000"
echo -e "   Airflow:    http://localhost:8080 (admin/admin)"
echo -e "   MLflow:     http://localhost:5000"
echo -e "   API Health: http://localhost:8081/health"
echo -e "==================================================="
echo -e " Streaming logs... Press [Ctrl+C] to stop."

docker compose logs -f
