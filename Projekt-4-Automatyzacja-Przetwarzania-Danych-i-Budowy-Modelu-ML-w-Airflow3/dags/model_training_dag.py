import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


def load_data(images_path, annotations_path, target_size=(224, 224)):
    images = []
    labels = []

    for breed_folder in os.listdir(images_path):
        breed_images_path = os.path.join(images_path, breed_folder)
        breed_annotations_path = os.path.join(annotations_path, breed_folder.replace("Images", "Annotation"))

        for image_file in os.listdir(breed_images_path):
            image_path = os.path.join(breed_images_path, image_file)
            annotation_file = image_file.replace('.jpg', '')  # Nazwa pliku bez rozszerzenia

            # Ładowanie adnotacji XML
            annotation_path = os.path.join(breed_annotations_path, annotation_file)
            try:
                tree = ET.parse(annotation_path)
                root = tree.getroot()

                # Wyciąganie bounding box i etykiety
                xmin = int(root.find('.//xmin').text)
                ymin = int(root.find('.//ymin').text)
                xmax = int(root.find('.//xmax').text)
                ymax = int(root.find('.//ymax').text)
                label = root.find('.//name').text

                # Wczytywanie i przycinanie obrazu
                image = cv2.imread(image_path)
                cropped_image = image[ymin:ymax, xmin:xmax]
                resized_image = cv2.resize(cropped_image, target_size)

                images.append(resized_image)
                labels.append(label)
            except Exception as e:
                print(f"Błąd przetwarzania pliku {annotation_path}: {e}")

    return np.array(images), np.array(labels)


def train_model():
    # Ścieżki danych
    images_path = '/tmp/split_dataset/train/images'
    annotations_path = '/tmp/split_dataset/train/annotations'

    # Ładowanie danych
    images, labels = load_data(images_path, annotations_path)

    # Konwertowanie etykiet do formatów liczbowych
    unique_labels = np.unique(labels)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_to_index[label] for label in labels])

    # Dzielenie na zestaw treningowy i walidacyjny
    X_train, X_val, y_train, y_val = train_test_split(images, y, test_size=0.2, random_state=42)

    # Tworzenie modelu
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(unique_labels), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Zamrażanie warstw modelu bazowego
    for layer in base_model.layers:
        layer.trainable = False

    # Kompilacja modelu
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Generatory danych
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = datagen.flow(X_train, y_train, batch_size=32)
    val_generator = ImageDataGenerator().flow(X_val, y_val, batch_size=32)

    # Trenowanie modelu
    history = model.fit(train_generator, epochs=10, validation_data=val_generator)

    # Ewaluacja modelu
    loss, accuracy = model.evaluate(val_generator)
    print(f"Validation Accuracy: {accuracy:.2f}")

    # Zapisywanie modelu
    model_save_path = "/tmp/dog_breed_model.h5"
    model.save(model_save_path)
    print(f"Model zapisano w: {model_save_path}")


# Definicja DAG-a
with DAG(
    dag_id='model_training_dag',
    start_date=datetime(2024, 10, 1),
    schedule_interval=None
) as dag:
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )
