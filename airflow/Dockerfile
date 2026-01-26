# airflow/Dockerfile

FROM apache/airflow:3.1.6

USER root
# Install Docker CLI so the DockerOperator can spin up containers
RUN apt-get update && \
    apt-get install -y docker.io && \
    rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER airflow

RUN pip install --no-cache-dir --upgrade docker

# Install Docker Provider
# RUN pip install apache-airflow-providers-docker==4.0.0 apache-airflow-providers-fab

RUN pip install --no-cache-dir apache-airflow-providers-docker==4.0.0 
RUN pip install apache-airflow-providers-http

ENTRYPOINT ["/entrypoint.sh"]
