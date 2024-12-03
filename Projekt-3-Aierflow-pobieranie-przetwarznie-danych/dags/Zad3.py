import logging
import requests
import os
# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log_google_sheets_api.txt"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()
api_key = 'AIzaSyCES-641xzQQRHVHGeh1iMmft6tJsLhuYs'
sheet_id = '1lJYX8jgR6ijCfu-wJW7_ki8sxWNJQNcOfN77tvB4Pyc'

url = f'https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}/values/A1:G10000?key={api_key}'  # Zakładamy, że dane są w pierwszych 1000 wierszach


# Funkcja do wgrywania danych do Google Sheets
def upload_to_google_sheets(dataframe, sheet_id, sheet_name, api_key):
    logger.info(f"Zapis danych do Google Sheets: {sheet_name}...")
    try:
        # Przygotowanie adresu URL dla arkusza
        url = f'https://sheets.googleapis.com/v4/spreadsheets/{sheet_id}/values/A1:G10000?key={api_key}'

        # Przygotowanie danych w formacie JSON
        data = {
            "range": f"{sheet_name}!A1",
            "majorDimension": "ROWS",
            "values": [dataframe.columns.tolist()] + dataframe.values.tolist()
        }

        # Wysłanie zapytania POST do API Google Sheets
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"}
        )

        # Sprawdzenie odpowiedzi
        if response.status_code == 200:
            logger.info(f"Dane zapisane w arkuszu: {sheet_name}.")
        else:
            logger.error(f"Błąd podczas zapisu danych. Kod odpowiedzi: {response.status_code}, Treść: {response.text}")
    except Exception as e:
        logger.error(f"Błąd podczas zapisywania danych do Google Sheets: {e}")



def load_data_with_paths(image_folder, annotation_folder):
    data = []
    for breed_folder in os.listdir(image_folder):
        breed_path = os.path.join(image_folder, breed_folder)
        annotation_path = os.path.join(annotation_folder, breed_folder)

        if os.path.isdir(breed_path):
            for image_file in os.listdir(breed_path):
                image_path = os.path.join(breed_path, image_file)
                annotation_file = os.path.join(annotation_path, image_file.replace('.jpg', ''))
                data.append({
                    "breed": breed_folder.split("-")[-1],
                    "image_path": image_path,
                    "annotation_path": annotation_file
                })
    return pd.DataFrame(data)



def split_data(data, test_size=0.3, random_state=42):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data



from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data_task(**kwargs):
    current_working_directory = os.getcwd()
    # print output to the console
    print(current_working_directory)
    image_folder = "/opt/airflow/dags/images/Images"  # Ścieżka do folderu z obrazami
    annotation_folder = "/opt/airflow/dags/annotations/Annotation"  # Ścieżka do folderu z metadanymi
    data = load_data_with_paths(image_folder, annotation_folder)
    logger.info(f"Dane załadowane: {data.head()}")
    return data.to_dict()  # Zwraca dane jako słownik

def split_data_task(**kwargs):
    ti = kwargs['ti']
    data_dict = ti.xcom_pull(task_ids='load_data')

    # Konwertuj dane do DataFrame
    data = pd.DataFrame.from_dict(data_dict)

    # Podziel dane na treningowe i testowe
    train_data, test_data = split_data(data)

    # Zwróć dane jako słowniki
    return train_data.to_dict(), test_data.to_dict()

# Funkcja do przygotowania danych i wgrania ich do Google Sheets
def upload_train_data_to_sheets(**kwargs):
    # Dane treningowe
    train_data = pd.DataFrame({
        "breed": ["Chihuahua", "Papillon"],
        "image_path": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
        "annotation_path": ["/path/to/annotation1.xml", "/path/to/annotation2.xml"]
    })

    # Parametry Google Sheets
    api_key = "AIzaSyCES-641xzQQRHVHGeh1iMmft6tJsLhuYs"  # Wstaw swój klucz API Google
    sheet_id = "143IPcDvI36XYZ0ioOMUWPcYpZPcpvaxBl7D0rEbPveM"  # Wstaw ID arkusza Google (znajdziesz w URL)
    sheet_name = "Train Data"  # Nazwa arkusza w Google Sheets

    # Zapis do Google Sheets
    upload_to_google_sheets(train_data, sheet_id, sheet_name, api_key)

def upload_test_data_to_sheets(**kwargs):
    # Dane testowe
    test_data = pd.DataFrame({
        "breed": ["Beagle", "Bulldog"],
        "image_path": ["/path/to/image3.jpg", "/path/to/image4.jpg"],
        "annotation_path": ["/path/to/annotation3.xml", "/path/to/annotation4.xml"]
    })

    # Parametry Google Sheets
    api_key = "AIzaSyCES-641xzQQRHVHGeh1iMmft6tJsLhuYs"  # Wstaw swój klucz API Google
    sheet_id = "1lJYX8jgR6ijCfu-wJW7_ki8sxWNJQNcOfN77tvB4Pyc"  # Wstaw ID arkusza Google (znajdziesz w URL)
    sheet_name = "ASIcW3"  # Nazwa arkusza w Google Sheets

    # Zapis do Google Sheets
    upload_to_google_sheets(test_data, sheet_id, sheet_name, api_key)


# Definicja DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 11, 19),
    'retries': 1,
}

dag = DAG(
    'upload_to_google_sheets_dag',
    default_args=default_args,
    description='DAG do wgrywania danych do Google Sheets',
    schedule_interval=None,  # Uruchamiaj ręcznie
)


load_data_operator = PythonOperator(
    task_id='load_data',
    python_callable=load_data_task,
    provide_context=True,
    dag=dag,
)

split_data_operator = PythonOperator(
    task_id='split_data',
    python_callable=split_data_task,
    provide_context=True,
    dag=dag,
)

# Taski w DAG-u
upload_train_task = PythonOperator(
    task_id='upload_train_data',
    python_callable=upload_train_data_to_sheets,
    dag=dag,
)

upload_test_task = PythonOperator(
    task_id='upload_test_data',
    python_callable=upload_test_data_to_sheets,
    dag=dag,
)




# Ustal kolejność zadań
load_data_operator >> split_data_operator >> [upload_test_task, upload_train_task]
