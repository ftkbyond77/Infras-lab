from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import os

HOST_MODEL_STORE = os.getenv("AIRFLOW_VAR_HOST_MODEL_STORE")

if not HOST_MODEL_STORE:
    raise ValueError("CRITICAL: AIRFLOW_VAR_HOST_MODEL_STORE is not set.")

default_args = {
    'owner': 'admin',
    # 'start_date': datetime(2025, 1, 1),
    'start_date': days_ago(1),
    'retries': 0,
}

dag = DAG(
    'mlops_cyclegan_v3',
    default_args=default_args,
    schedule=None,
    catchup=False
)

# 1. Train Model Task
train_task = DockerOperator(
    task_id='train_model_container',
    image='infras-lab-inference:latest', 
    
    api_version='auto',
    auto_remove=True,
    network_mode='infras-lab_default',
    docker_url='unix://var/run/docker.sock',
    command='python model_training.py',
    environment={
        'MLFLOW_URI': 'http://mlflow:5000',
        # 'AIRFLOW__CORE__EXECUTION_API_SERVER_URL': 'http://airflow-api-server:8080/execution/' 
    },
    mount_tmp_dir=False,
    volumes=[f'{HOST_MODEL_STORE}/candidates:/outputs'],
    dag=dag,
)

# 2. Deploy Task
# This command runs inside the Airflow Scheduler container, so it sees the 'internal' path
deploy_task = BashOperator(
    task_id='promote_model',
    bash_command='mv /opt/airflow/model_store/candidates/cyclegan_candidate.onnx /opt/airflow/model_store/latest/cyclegan.onnx',
    dag=dag
)

train_task >> deploy_task