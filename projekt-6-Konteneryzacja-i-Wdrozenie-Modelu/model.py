import os
import cv2
import pandas as pd
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, concatenate
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D

# Funkcja do wczytywania metadanych z plików XML
def parse_metadata(annotation_path, image_folder):
    data = []
    for breed_folder in os.listdir(annotation_path):
        breed_path = os.path.join(annotation_path, breed_folder)
        if os.path.isdir(breed_path):
            for file in os.listdir(breed_path):
                file_path = os.path.join(breed_path, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Parsowanie nazwy rasy i bounding box (jeśli istnieje)
                    name_tag_start = content.find('<name>') + 6
                    name_tag_end = content.find('</name>')
                    breed_name = content[name_tag_start:name_tag_end]

                    xmin_start = content.find('<xmin>') + 6
                    xmin_end = content.find('</xmin>')
                    ymin_start = content.find('<ymin>') + 6
                    ymin_end = content.find('</ymin>')
                    xmax_start = content.find('<xmax>') + 6
                    xmax_end = content.find('</xmax>')
                    ymax_start = content.find('<ymax>') + 6
                    ymax_end = content.find('</ymax>')

                    bbox = None
                    if xmin_start > 6:
                        bbox = [
                            int(content[xmin_start:xmin_end]),
                            int(content[ymin_start:ymin_end]),
                            int(content[xmax_start:xmax_end]),
                            int(content[ymax_start:ymax_end]),
                        ]

                    # Ścieżka do obrazu
                    image_path = os.path.join(image_folder, breed_folder, f"{file}.jpg")

                    # Dodanie wiersza do danych
                    data.append({
                        "breed": breed_name,
                        "image_path": image_path,
                        "bbox": bbox
                    })
    return pd.DataFrame(data)


# Funkcja do przetwarzania zdjęć
def load_and_process_images(metadata_df, target_size=(224, 224)):
    images = []
    processed_metadata = []
    for index, row in metadata_df.iterrows():
        image_path = row['image_path']
        try:
            # Wczytanie i zmiana rozmiaru obrazu
            img = load_img(image_path, target_size=target_size)
            img_array = img_to_array(img)
            images.append(img_array)

            # Przekazanie pozostałych danych
            processed_metadata.append({
                "breed": row['breed'],
                "bbox": row['bbox']
            })
        except Exception as e:
            print(f"Problem z przetwarzaniem obrazu: {image_path}, {e}")
    return images, pd.DataFrame(processed_metadata)


# Ścieżki do danych
annotations_path = os.path.join("D:\\ASI\\annotations\\Annotation")
images_path = os.path.join("D:\\ASI\\images\\Images")

# Parsowanie metadanych
metadata_df = parse_metadata(annotations_path, images_path)

# Przetwarzanie zdjęć i metadanych
images, processed_metadata = load_and_process_images(metadata_df)

# Wyświetlenie wyników
print(f"Liczba przetworzonych obrazów: {len(images)}")
print(f"Liczba przetworzonych rekordów metadanych: {processed_metadata.shape[0]}")
processed_metadata.head()


# Normalizacja obrazów i przygotowanie danych
print("Normalizowanie obrazów...")
images_np = np.array(images, dtype="float32") / 255.0  # Normalizacja obrazów (zakres 0-1)
print("Przygotowywanie etykiet i metadanych...")

# Kodowanie etykiet klas (nazwy ras)
labels = processed_metadata["breed"].values
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)  # Kodowanie kategorii na wektory

# Przygotowanie bounding boxów (z domyślnymi wartościami, jeśli brak danych)
bounding_boxes = np.array([bbox if bbox else [0, 0, 0, 0] for bbox in processed_metadata["bbox"]])

# Podział na dane treningowe i testowe
X_train_imgs, X_test_imgs, X_train_boxes, X_test_boxes, y_train, y_test = train_test_split(
    images_np, bounding_boxes, labels_categorical, test_size=0.2, random_state=42
)

print(f"Liczba obrazów treningowych: {X_train_imgs.shape[0]}")
print(f"Liczba obrazów testowych: {X_test_imgs.shape[0]}")

# Budowanie modelu CNN

# Pretrenowany model MobileNetV2
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Wejście dla obrazu z poprawioną nazwą
image_input = Input(shape=(224, 224, 3), name="input_layer")  # Zmieniono nazwę na "input_layer"
x = base_model(image_input)
x = GlobalAveragePooling2D()(x)  # Globalne uśrednianie cech
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
image_features = Dense(64, activation="relu")(x)

# Model przetwarzający bounding boxy
bbox_input = Input(shape=(4,), name="bbox_input")
y = Dense(32, activation="relu")(bbox_input)
bbox_features = Dense(16, activation="relu")(y)

# Łączenie cech
combined = concatenate([image_features, bbox_features])
z = Dense(128, activation="relu")(combined)
z = Dropout(0.5)(z)
z = Dense(len(label_encoder.classes_), activation="softmax")(z)

# Model końcowy
model = Model(inputs=[image_input, bbox_input], outputs=z)

# Zamrożenie warstw pretrenowanego modelu
for layer in base_model.layers:
    layer.trainable = False

# Kompilacja modelu
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Trenowanie modelu
history = model.fit(
    {"input_layer": X_train_imgs, "bbox_input": X_train_boxes},  # Poprawione klucze danych wejściowych
    y_train,
    validation_data=(
        {"input_layer": X_test_imgs, "bbox_input": X_test_boxes},  # Poprawione klucze danych wejściowych
        y_test,
    ),
    epochs=10,
    batch_size=32
)


# Ocena modelu na danych testowych
results = model.evaluate(
    {"input_layer": X_test_imgs, "bbox_input": X_test_boxes},
    y_test,
    batch_size=32
)

print(f"Test Loss: {results[0]}")
print(f"Test Accuracy: {results[1]}")

# Przewidywanie klas na danych testowych
predictions = model.predict(
    {"input_layer": X_test_imgs, "bbox_input": X_test_boxes},
    batch_size=32
)

# Pobranie indeksów klas o najwyższych wartościach predykcji
predicted_classes = np.argmax(predictions, axis=1)

# Dekodowanie indeksów na nazwy ras
predicted_labels = label_encoder.inverse_transform(predicted_classes)

# Wyświetlenie kilku przykładowych predykcji
for i in range(5):
    print(f"Predicted: {predicted_labels[i]}, Actual: {label_encoder.inverse_transform([np.argmax(y_test[i])])[0]}")


import matplotlib.pyplot as plt

# Wyświetlenie kilku obrazów z przewidywaniami
num_images_to_show = 5
plt.figure(figsize=(15, 10))
for i in range(num_images_to_show):
    plt.subplot(1, num_images_to_show, i + 1)
    plt.imshow(X_test_imgs[i])
    plt.title(f"Predicted: {predicted_labels[i]}\nActual: {label_encoder.inverse_transform([np.argmax(y_test[i])])[0]}")
    plt.axis('off')
plt.show()


# Zapisanie modelu do pliku HDF5
model.save("dog_breed_model40.h5")
print("Model zapisany jako dog_breed_model40.h5")

import pickle
# Zapisanie LabelEncoder
with open("label_encoder.pkl", "wb") as file:
    pickle.dump(label_encoder, file)


# # Zapisanie modelu w formacie SavedModel
# model.save("dog_breed_model_saved")
# print("Model zapisany jako dog_breed_model_saved")



# Wizualizacja wyników treningu
print("Wizualizacja wyników...")
plt.figure(figsize=(12, 4))

# Wykres strat
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss - Train')
plt.plot(history.history['val_loss'], label='Loss - Validation')
plt.title('Wykres strat')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()

# Wykres dokładności
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy - Train')
plt.plot(history.history['val_accuracy'], label='Accuracy - Validation')
plt.title('Wykres dokładności')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()

plt.show()
