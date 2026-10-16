from datetime import datetime, timedelta
import os
import sys
from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.ingest import run_ingestion
from src.etl_pipeline import run_etl
from src.train_mlflow import run_ml_pipeline
from src.drift_monitor import check_feature_drift

default_args = {
    "owner": "aishwarya",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="student_performance_data_pipeline",
    default_args=default_args,
    description="End-to-End Academic Analytics & MLOps Pipeline",
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["postgresql", "mlflow", "mlops", "data-engineering"],
) as dag:

    task_ingest = PythonOperator(
        task_id="ingest_uci_student_data",
        python_callable=run_ingestion,
    )

    task_etl = PythonOperator(
        task_id="transform_validate_and_load_postgres",
        python_callable=run_etl,
    )

    task_train = PythonOperator(
        task_id="train_models_and_log_mlflow",
        python_callable=run_ml_pipeline,
    )

    task_monitor = PythonOperator(
        task_id="check_data_drift_and_retraining",
        python_callable=check_feature_drift,
    )

    task_ingest >> task_etl >> task_train >> task_monitor