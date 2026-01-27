from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount, DeviceRequest
# from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.standard.operators.bash import BashOperator
# from airflow.operators.bash import BashOperator
from airflow.providers.http.operators.http import HttpOperator
from airflow.sdk.bases.hook import BaseHook
from datetime import datetime, timezone

import os
import shutil
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
HOST_MODEL_STORE = os.getenv("AIRFLOW_VAR_HOST_MODEL_STORE", "/tmp/model_store")


AIRFLOW_DATA_PATH = "/opt/airflow/model_store/data"
RAW_PATH = f"{AIRFLOW_DATA_PATH}/raw"
PROCESSED_PATH = f"{AIRFLOW_DATA_PATH}/processed"

default_args = {
    'owner': 'admin',
    'start_date': datetime(2025, 1, 1, tzinfo=timezone.utc),
    'retries': 0,
}


# ------------------------------------------------------------------------------
# DAG 1: DATA PIPELINE
# ------------------------------------------------------------------------------
def _preprocess_data(**kwargs):
    processed_path = Path(PROCESSED_PATH)
    raw_path = Path(RAW_PATH)
    
    logger.info(f"Starting preprocessing: raw={raw_path}, processed={processed_path}")

    if processed_path.exists():
        logger.info("Clearing existing processed directory contents.")
        for item in list(processed_path.iterdir()):
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to remove {item}: {e}")
    
    processed_path.mkdir(parents=True, exist_ok=True)

    # Small Retry if copy fails due to race
    for attempt in range(3):
        try:
            shutil.copytree(raw_path, processed_path, copy_function=shutil.copy, dirs_exist_ok=True)
            logger.info(f"Successfully copied data from {raw_path} to {processed_path}")
            break
        except Exception as e:
            logger.warning(f"Copy attempt {attempt+1} failed: {e}")
            if attempt == 2:
                raise
            time.sleep(1)

    print(f"Data processed from {RAW_PATH} to {PROCESSED_PATH}")


with DAG('01_data_pipeline', default_args=default_args, schedule='@daily', catchup=False) as dag_data:
    preprocess_task = PythonOperator(
        task_id='preprocess_images',
        python_callable=_preprocess_data
    )


# ------------------------------------------------------------------------------
# DAG 2: TRAINING PIPELINE
# ------------------------------------------------------------------------------
with DAG('02_training_pipeline', default_args=default_args, schedule=None, catchup=False) as dag_train:
    
    # Runs the training script inside the PyTorch container
    train_model = DockerOperator(
        task_id='train_model_container',
        image='infras-lab-training:latest', 
        api_version='auto',
        auto_remove="success",
        network_mode='infras-lab_default', # Must match docker-compose network name
        docker_url='unix://var/run/docker.sock',
        command='python model_training.py',

        device_requests=[
            DeviceRequest(count=-1, capabilities=[['gpu']])
        ],

        environment={
            'MLFLOW_URI': 'http://mlflow:5000',
            # Map internal paths to where we mounted the volume below
            'BASE_DATA_DIR': '/outputs/data/processed',
            'OUTPUT_MODEL_DIR': '/outputs/candidates'
        },
        mounts=[
            Mount(
                source=HOST_MODEL_STORE,    # Host path
                target='/outputs',          # container path
                type='bind'
            )
        ],
        mount_tmp_dir=False, # disable temp mount
        force_pull=False,
    )


# ------------------------------------------------------------------------------
# DAG 3: DEPLOYMENT PIPELINE (CI/CD)
# ------------------------------------------------------------------------------
def _compare_models(**kwargs):
    # Logic: Check MLflow metrics. For now, always promote.
    return 'promote_model'

# def _compare_models():
#     import mlflow

#     client = mlflow.tracking.MlflowClient()
#     runs = client.search_runs(experiment_ids=["1"], order_by=["metrics.fid ASC"])

#     best_new = runs[0].data.metrics["fid"]
#     prod_fid = load_prod_metric()

#     if best_new < prod_fid:
#         return "promote_model"
#     else:
#         return "reject_model"


with DAG('03_deploy_pipeline', default_args=default_args, schedule=None, catchup=False) as dag_deploy:
    
    check_quality = BranchPythonOperator(
        task_id='validate_candidate',
        python_callable=_compare_models
    )

    # Move candidate ONNX to 'latest' folder where Go service looks
    promote_model = BashOperator(
        task_id='promote_model',
        bash_command='cp /opt/airflow/model_store/candidates/cyclegan_candidate.onnx /opt/airflow/model_store/latest/cyclegan.onnx'
    )

    # Trigger Hot-Reload on Go Service
    # Note: Requires Connection 'inference_server' (Host: inference, Port: 8080)
    reload_service = HttpOperator(
        task_id='trigger_hot_reload',
        http_conn_id='inference_server', 
        endpoint='reload',
        method='POST',
        headers={"Content-Type": "application/json"},
    )

    reject_model = BashOperator(
        task_id='reject_model',
        bash_command='echo "Model rejected"'
    )

    check_quality >> [promote_model, reject_model]
    promote_model >> reload_service