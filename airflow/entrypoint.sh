#!/bin/bash
#!/bin/bash
set -e
airflow db migrate || true
exec "$@"
# exec /entrypoint "$@"


# --- Password ---
# /opt/airflow/simple_auth_manager_passwords.json.generated

# --- Check via ---
# docker compose exec airflow-api-server cat /opt/airflow/simple_auth_manager_passwords.json.generated
