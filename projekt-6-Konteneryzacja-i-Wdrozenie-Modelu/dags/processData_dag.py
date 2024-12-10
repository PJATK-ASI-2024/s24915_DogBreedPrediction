from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

default_args = {
    'owner': 'airflow',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

def preprocess_data():
    # Funkcja przetwarzająca dane
    print("Processing Data...")

with DAG(
    dag_id="dog_breed_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    task1 = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )

    task1
