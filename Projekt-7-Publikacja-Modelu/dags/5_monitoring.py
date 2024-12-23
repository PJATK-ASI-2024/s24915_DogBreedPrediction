from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from datetime import datetime
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle

# Funkcje do monitoringu

def load_model_and_data(**kwargs):
    # Wczytanie modelu i danych
    model_path = kwargs['model_path']
    test_data_path = kwargs['test_data_path']
    label_encoder_path = kwargs['label_encoder_path']

    model = tf.keras.models.load_model(model_path)
    test_data = np.load(test_data_path, allow_pickle=True)

    with open(label_encoder_path, 'rb') as file:
        label_encoder = pickle.load(file)

    return model, test_data, label_encoder

def analyze_model_performance(model, test_data, **kwargs):
    X_test_imgs = test_data['X_test_imgs']
    X_test_boxes = test_data['X_test_boxes']
    y_test = test_data['y_test']

    # Ocena modelu
    results = model.evaluate(
        {"input_layer": X_test_imgs, "bbox_input": X_test_boxes},
        y_test,
        batch_size=kwargs['batch_size']
    )

    metrics = {
        'loss': results[0],
        'accuracy': results[1]
    }

    # Sprawdzenie jakości modelu
    if metrics['accuracy'] < kwargs['accuracy_threshold']:
        raise ValueError(f"Accuracy {metrics['accuracy']} poniżej progu {kwargs['accuracy_threshold']}.")

    return metrics

def run_model_tests(model, test_data, label_encoder, **kwargs):
    X_test_imgs = test_data['X_test_imgs']
    X_test_boxes = test_data['X_test_boxes']
    y_test = test_data['y_test']

    # Przewidywanie
    predictions = model.predict(
        {"input_layer": X_test_imgs, "bbox_input": X_test_boxes},
        batch_size=kwargs['batch_size']
    )

    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    # Stworzenie przykładowego raportu
    sample_report = []
    for i in range(min(10, len(predicted_classes))):
        sample_report.append({
            'predicted': label_encoder.inverse_transform([predicted_classes[i]])[0],
            'actual': label_encoder.inverse_transform([true_classes[i]])[0]
        })

    return sample_report

def generate_quality_report(metrics, sample_report, **kwargs):
    report_path = kwargs['report_path']

    with open(report_path, 'w') as report_file:
        report_file.write("Model Quality Report\n")
        report_file.write("====================\n\n")
        report_file.write(f"Loss: {metrics['loss']}\n")
        report_file.write(f"Accuracy: {metrics['accuracy']}\n\n")
        report_file.write("Sample Predictions:\n")
        for entry in sample_report:
            report_file.write(f"Predicted: {entry['predicted']}, Actual: {entry['actual']}\n")

    return report_path

# Parametry DAG
params = {
    'model_path': 'dog_breed_model.h5',
    'test_data_path': 'test_data.npy',
    'label_encoder_path': 'label_encoder.pkl',
    'report_path': 'quality_report.txt',
    'batch_size': 32,
    'accuracy_threshold': 0.85,
    'email_on_failure': 'your_email@example.com'
}

with DAG(
    dag_id='model_monitoring_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None
) as dag:

    load_model_and_data_task = PythonOperator(
        task_id='load_model_and_data',
        python_callable=load_model_and_data,
        op_kwargs=params,
        provide_context=True
    )

    analyze_model_performance_task = PythonOperator(
        task_id='analyze_model_performance',
        python_callable=analyze_model_performance,
        op_kwargs=params,
        provide_context=True
    )

    run_model_tests_task = PythonOperator(
        task_id='run_model_tests',
        python_callable=run_model_tests,
        op_kwargs=params,
        provide_context=True
    )

    generate_quality_report_task = PythonOperator(
        task_id='generate_quality_report',
        python_callable=generate_quality_report,
        op_kwargs=params,
        provide_context=True
    )

    send_email_on_failure_task = EmailOperator(
        task_id='send_email_on_failure',
        to=params['email_on_failure'],
        subject='Model Monitoring Alert',
        html_content='Monitoring failed. Please check the logs for details.',
        trigger_rule='one_failed'
    )

    # Kolejność zadań
    load_model_and_data_task >> analyze_model_performance_task >> run_model_tests_task >> generate_quality_report_task
    analyze_model_performance_task >> send_email_on_failure_task
