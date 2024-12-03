from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email_operator import EmailOperator
from datetime import datetime
import os
import shutil
import gdown
import zipfile
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

# Define the critical threshold for model quality
CRITICAL_ACCURACY_THRESHOLD = 0.80
MODEL_PATH = '/path/to/model.pkl'  # Path to the saved model
TEST_DATA_PATH = '/path/to/test_data.npz'  # Path to the test dataset
EMAIL_RECIPIENTS = ['jaklip3322@gmail.com']

# Function to evaluate model quality
def evaluate_model_quality():
    # Load test data
    data = np.load(TEST_DATA_PATH)
    X_test, y_test = data['X'], data['y']

    # Load the trained model
    model = joblib.load(MODEL_PATH)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")

    # Return the result of the evaluation
    return accuracy

# Function to check if model tests pass
def run_model_tests():
    # Placeholder for model tests
    # Return True if all tests pass, False otherwise
    tests_passed = True  # Replace with real test logic
    return tests_passed

# Define the DAG
with DAG(
        dag_id='model_quality_monitoring_dag',
        start_date=datetime(2024, 10, 1),
        schedule_interval='@daily'
) as dag:

    # Task to evaluate model quality
    evaluate_model_task = PythonOperator(
        task_id='evaluate_model_quality',
        python_callable=evaluate_model_quality
    )

    # Task to run model tests
    run_model_tests_task = PythonOperator(
        task_id='run_model_tests',
        python_callable=run_model_tests
    )

    # Task to send an alert email
    def send_alert_email(**context):
        accuracy = context['task_instance'].xcom_pull(task_ids='evaluate_model_quality')
        tests_passed = context['task_instance'].xcom_pull(task_ids='run_model_tests')

        subject = f"Model Quality Alert - {MODEL_PATH.split('/')[-1]}"
        body = f"Model: {MODEL_PATH}\nCurrent accuracy: {accuracy}\nCritical threshold: {CRITICAL_ACCURACY_THRESHOLD}\n"

        if not tests_passed:
            body += "Some tests did not pass successfully."

        email_task = EmailOperator(
            task_id='send_email',
            to=EMAIL_RECIPIENTS,
            subject=subject,
            html_content=body
        )
        email_task.execute(context=context)

    send_alert_email_task = PythonOperator(
        task_id='send_alert_email',
        python_callable=send_alert_email,
        provide_context=True
    )

    # Set up task dependencies
    evaluate_model_task >> run_model_tests_task >> send_alert_email_task
