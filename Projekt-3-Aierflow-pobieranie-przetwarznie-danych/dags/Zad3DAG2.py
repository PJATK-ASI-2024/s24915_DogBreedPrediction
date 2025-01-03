from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Parametry Google Sheets
API_KEY = "AIzaSyCES-641xzQQRHVHGeh1iMmft6tJsLhuYs"
SHEET_ID = "1lJYX8jgR6ijCfu-wJW7_ki8sxWNJQNcOfN77tvB4Pyc"
SHEET_NAME = "Train Data"

# Funkcja do pobierania danych z Google Sheets
def fetch_data_from_sheets():
    url = f'https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}/values/{SHEET_NAME}?key={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        values = response.json().get('values')
        columns = values[0]
        data = pd.DataFrame(values[1:], columns=columns)
        logger.info("Dane pobrane z Google Sheets.")
        return data
    else:
        logger.error("Błąd podczas pobierania danych z Google Sheets.")
        raise Exception("Nie udało się pobrać danych z Google Sheets.")

# Funkcja do czyszczenia danych
def clean_data(data):
    logger.info("Rozpoczynanie czyszczenia danych.")
    # Usuń wartości brakujące
    data = data.dropna()
    # Usuń duplikaty
    data = data.drop_duplicates()
    logger.info("Czyszczenie danych zakończone.")
    return data

# Funkcja do standaryzacji i normalizacji danych
def standardize_and_normalize(data):
    logger.info("Rozpoczynanie standaryzacji i normalizacji.")
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    if not numeric_columns.empty:
        # Standaryzacja
        scaler = StandardScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

        # Normalizacja
        normalizer = MinMaxScaler()
        data[numeric_columns] = normalizer.fit_transform(data[numeric_columns])
    logger.info("Standaryzacja i normalizacja zakończona.")
    return data

# Funkcja do zapisywania danych do Google Sheets
def save_data_to_sheets(data):
    logger.info("Zapisywanie przetworzonych danych do Google Sheets.")
    url = f'https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}/values/{SHEET_NAME}!A1:append?key={API_KEY}'
    data_payload = {
        "range": f"{SHEET_NAME}!A1",
        "majorDimension": "ROWS",
        "values": [data.columns.tolist()] + data.values.tolist()
    }
    response = requests.post(url, json=data_payload, headers={"Content-Type": "application/json"})
    if response.status_code == 200:
        logger.info("Dane zapisane do Google Sheets.")
    else:
        logger.error(f"Błąd podczas zapisu danych. Kod odpowiedzi: {response.status_code}, Treść: {response.text}")
        raise Exception("Nie udało się zapisać danych do Google Sheets.")

# Definicja DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 11, 20),
    'retries': 1,
}

dag = DAG(
    'data_processing_dag',
    default_args=default_args,
    description='DAG do przetwarzania danych',
    schedule_interval=None,
)

# Taski
fetch_data_task = PythonOperator(
    task_id='fetch_data',
    python_callable=fetch_data_from_sheets,
    dag=dag,
)

clean_data_task = PythonOperator(
    task_id='clean_data',
    python_callable=lambda **kwargs: clean_data(fetch_data_task.output),
    provide_context=True,
    dag=dag,
)

process_data_task = PythonOperator(
    task_id='process_data',
    python_callable=lambda **kwargs: standardize_and_normalize(clean_data_task.output),
    provide_context=True,
    dag=dag,
)

save_data_task = PythonOperator(
    task_id='save_data',
    python_callable=lambda **kwargs: save_data_to_sheets(process_data_task.output),
    provide_context=True,
    dag=dag,
)

# Kolejność tasków
fetch_data_task >> clean_data_task >> process_data_task >> save_data_task
