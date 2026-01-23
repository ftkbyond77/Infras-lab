# Infras-lab

Test Docker Locally:
```
# 1. Build the image
cd inferences
docker build -t my-cyclegan-inference .

# 2. Prepare local folders for testing
mkdir -p /tmp/data/model /tmp/data/io
cp ../model-path/cyclegan_horse2zebra.onnx /tmp/data/model/
cp The-Horses-Personality.jpg /tmp/data/io/horse.jpg

# 3. Run the container
docker run --rm \
  -v /tmp/data/model:/data/model \
  -v /tmp/data/io:/data/io \
  -e MODEL_PATH="/data/model/cyclegan_horse2zebra.onnx" \
  -e INPUT_IMAGE="/data/io/horse.jpg" \
  -e OUTPUT_IMAGE="/data/io/zebra_out.jpg" \
  my-cyclegan-inference
```

Check Files
```
# List the files to confirm it exists
ls -l /tmp/data/io/

# Copy it to your Desktop (so you can easily open and see it)
cp /tmp/data/io/zebra_out.jpg ~/Desktop/zebra_out.jpg
```

Test Kubernetes (Minikube (Local)):
```
# Load image into minikube
minikube image load my-cyclegan-inference

# Apply the job
kubectl apply -f k8s-inference-job.yaml

# Check logs
kubectl logs job/cyclegan-inference-job
```


Cleanup
```
# 1. Delete the Kubernetes Job
kubectl delete -f k8s-inference-job.yaml

# 2. Stop amd Delete Minikube
minikube stop
minikube delete

# 3. Cleanup Docker System
docker rmi my-cyclegan-inference:latest
docker system prune -f

# 4. Clear Temporary Data (/tmp)
rm -rf /tmp/data
```


Next.js (Front-End System)
```
1. Inference Model Runtime
docker build -t horse2zebra-app .
docker run -p 8080:8080 horse2zebra-app

2. cd frontend 
if not exists:
-> npx create-next-app@latest frontend
else:
-> npm run dev

3. Open Localhost:3000
```


Run By Shell (start.sh)
```
chmod +x start.sh

./start.sh
```

============== DATABASE SECTION (PostgreSQL) ==============
```
# Switch to the postgres user
sudo -i -u postgres

# Enter the Postgres shell
psql

# --- Inside SQL Shell ---
# 1. Create the database
CREATE DATABASE airflow_db;

# 2. Create the user (password: 'airflow_pass')
CREATE USER airflow_user WITH PASSWORD 'airflow_pass';

# 3. Grant privileges
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow_user;

# 4. (Specific to Postgres 15+) Grant schema usage
\c airflow_db
GRANT ALL ON SCHEMA public TO airflow_user;

# Exit
\q

```

Airflow Metadata USER
```
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@gmail.com \
    --password admin
```

Run Airflow (Apache Airflow 3 ++)
```
airflow scheduler &
airflow api-server -p 8081 &

For Check Process
ps aux | grep airflow

For Check Port
lsof -i :8081

Kill (Process)
pkill -f "<process_name>"

Kill (Port)
kill <process_number> 
ex. kill 21390
```

CREATE AIRFLOW DAG
```
mkdir -p ~/airflow/dags
nano ~/airflow/dags/cyclegan_pipeline.py

```


Current Flow
```
Terminal 1: mlflow ui
Terminal 2: airflow scheduler &
            airflow api-server -p 8081 &
Terminal 3: python3 model-training/model_training.py
  Option B: Production Traigger (Via Airflow) If Option A works
            airflow dags trigger horse2zebra_training_pipeline

Check
Apache Airflow: Localhost:8081
MLflow: Localhost:5000
```