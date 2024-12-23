from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

# Funkcje pomocnicze

def generate_dockerfile(**kwargs):
    dockerfile_content = f"""
    FROM python:3.9-slim

    WORKDIR /app

    # Instalacja zależności
    COPY requirements.txt ./
    RUN pip install --no-cache-dir -r requirements.txt

    # Kopiowanie aplikacji
    COPY . ./

    # Uruchomienie aplikacji
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    """

    with open('Dockerfile', 'w') as f:
        f.write(dockerfile_content)


def create_requirements(**kwargs):
    requirements_content = """
    fastapi
    uvicorn
    tensorflow
    numpy
    pydantic
    """

    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)

# Parametry DAG
params = {
    'image_name': 'your_dockerhub_username/dog_breed_api',
    'dockerhub_username': 'your_dockerhub_username',
    'dockerhub_password': 'your_dockerhub_password',
    'api_files': ['main.py', 'dog_breed_model40.h5', 'label_encoder.pkl']
}

with DAG(
    dag_id='contenerysation_and_api',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None
) as dag:

    # Zadanie 1: Generowanie pliku Dockerfile
    generate_dockerfile_task = PythonOperator(
        task_id='generate_dockerfile',
        python_callable=generate_dockerfile
    )

    # Zadanie 2: Generowanie pliku requirements.txt
    create_requirements_task = PythonOperator(
        task_id='create_requirements',
        python_callable=create_requirements
    )

    # Zadanie 3: Budowanie obrazu Dockera
    build_docker_image_task = BashOperator(
        task_id='build_docker_image',
        bash_command=f'docker build -t {params["image_name"]} .'
    )

    # Zadanie 4: Publikacja obrazu Dockera na Docker Hub
    publish_docker_image_task = BashOperator(
        task_id='publish_docker_image',
        bash_command=f'docker login -u {params["dockerhub_username"]} -p {params["dockerhub_password"]} && docker push {params["image_name"]}'
    )

    # Zadanie 5: Sprawdzenie obrazu Dockera (opcjonalne)
    verify_docker_image_task = BashOperator(
        task_id='verify_docker_image',
        bash_command=f'docker run --rm -p 8000:8000 {params["image_name"]}'
    )

    # Kolejność zadań
    generate_dockerfile_task >> create_requirements_task >> build_docker_image_task >> publish_docker_image_task >> verify_docker_image_task
