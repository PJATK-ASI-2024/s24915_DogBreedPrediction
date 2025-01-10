from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pickle
from pydantic import BaseModel

# Ścieżki do modelu i encoder'a
MODEL_PATH = "dog_breed_model40.h5"
ENCODER_PATH = "label_encoder.pkl"

# Załadowanie modelu i LabelEncoder
model = load_model(MODEL_PATH)
with open(ENCODER_PATH, "rb") as file:
    label_encoder = pickle.load(file)

# Utworzenie aplikacji FastAPI
app = FastAPI(title="Dog Breed Prediction API")

# Funkcja do przewidywania
def predict_new_image(image_path, bbox=None):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if bbox is None:
        bbox = [0, 0, 0, 0]
    bbox_array = np.array([bbox])

    prediction = model.predict({"input_layer": img_array, "bbox_input": bbox_array})
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)

    return predicted_label[0]

# Punkt końcowy do przewidywania
@app.post("/predict/")
async def predict(file: UploadFile = File(...), xmin: int = Form(0), ymin: int = Form(0), xmax: int = Form(0), ymax: int = Form(0)):
    try:
        # Zapisanie przesłanego pliku tymczasowo
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Przygotowanie bounding boxa
        bbox = [xmin, ymin, xmax, ymax]

        # Przewidywanie
        predicted_label = predict_new_image(file_path, bbox)

        # Usunięcie tymczasowego pliku
        import os
        os.remove(file_path)

        # Zwrot wyniku
        return JSONResponse({"predicted_label": predicted_label})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
