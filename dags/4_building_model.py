from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
import pickle


# Funkcje do pipeline

def load_data(**kwargs):
    # Simulacja pobrania danych z chmury (Google Sheets lub inne)
    metadata_path = kwargs['metadata_path']
    image_path = kwargs['image_path']
    # Ładowanie metadanych
    metadata_df = pd.read_csv(metadata_path)
    images = np.load(image_path)  # Zakładamy, że obrazy są już przygotowane

    return metadata_df, images


def preprocess_data(metadata_df, images, **kwargs):
    # Kodowanie etykiet klas (nazwy ras)
    labels = metadata_df['breed'].values
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)  # Kodowanie kategorii na wektory

    # Przygotowanie bounding boxów
    bounding_boxes = np.array([bbox if bbox else [0, 0, 0, 0] for bbox in metadata_df['bbox']])

    # Podział na dane treningowe i testowe
    X_train_imgs, X_test_imgs, X_train_boxes, X_test_boxes, y_train, y_test = train_test_split(
        images, bounding_boxes, labels_categorical, test_size=0.2, random_state=42
    )

    return X_train_imgs, X_test_imgs, X_train_boxes, X_test_boxes, y_train, y_test, label_encoder


def train_model(X_train_imgs, X_train_boxes, y_train, **kwargs):
    # Budowanie modelu
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    image_input = Input(shape=(224, 224, 3), name="input_layer")
    x = base_model(image_input)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    image_features = Dense(64, activation="relu")(x)

    bbox_input = Input(shape=(4,), name="bbox_input")
    y = Dense(32, activation="relu")(bbox_input)
    bbox_features = Dense(16, activation="relu")(y)

    combined = concatenate([image_features, bbox_features])
    z = Dense(128, activation="relu")(combined)
    z = Dropout(0.5)(z)
    z = Dense(y_train.shape[1], activation="softmax")(z)

    model = Model(inputs=[image_input, bbox_input], outputs=z)

    # Kompilacja modelu
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Trenowanie modelu
    history = model.fit(
        {"input_layer": X_train_imgs, "bbox_input": X_train_boxes},
        y_train,
        epochs=kwargs['epochs'],
        batch_size=kwargs['batch_size'],
        validation_split=0.2
    )

    return model, history


def evaluate_model(model, X_test_imgs, X_test_boxes, y_test, **kwargs):
    results = model.evaluate(
        {"input_layer": X_test_imgs, "bbox_input": X_test_boxes},
        y_test,
        batch_size=kwargs['batch_size']
    )
    return results


def save_model_and_report(model, history, results, label_encoder, **kwargs):
    # Zapis modelu
    model.save(kwargs['model_path'])
    with open(kwargs['label_encoder_path'], 'wb') as file:
        pickle.dump(label_encoder, file)

    # Zapis raportu
    with open(kwargs['report_path'], 'w') as report_file:
        report_file.write(f"Test Loss: {results[0]}\n")
        report_file.write(f"Test Accuracy: {results[1]}\n")


# Parametry DAG
params = {
    'metadata_path': 'path/to/metadata.csv',
    'image_path': 'path/to/images.npy',
    'model_path': 'dog_breed_model40v1.h5',
    'label_encoder_path': 'label_encoder.pkl',
    'report_path': 'model_report.txt',
    'epochs': 10,
    'batch_size': 32
}

with DAG(
        dag_id='dog_breed_model_pipeline',
        start_date=datetime(2024, 1, 1),
        schedule_interval=None
) as dag:
    load_data_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
        op_kwargs=params,
        provide_context=True
    )

    preprocess_data_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
        provide_context=True
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        op_kwargs=params,
        provide_context=True
    )

    evaluate_model_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        op_kwargs=params,
        provide_context=True
    )

    save_model_and_report_task = PythonOperator(
        task_id='save_model_and_report',
        python_callable=save_model_and_report,
        op_kwargs=params,
        provide_context=True
    )

    # Kolejność zadań
    load_data_task >> preprocess_data_task >> train_model_task >> evaluate_model_task >> save_model_and_report_task
