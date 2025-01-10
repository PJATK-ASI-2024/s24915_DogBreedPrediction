# Dog Breed Prediction API

Aplikacja REST API służy do przewidywania rasy psa na podstawie przesłanego zdjęcia. API wykorzystuje model wytrenowany za pomocą TensorFlow i jest opakowane w kontener Docker.

## Funkcje

- Przewidywanie rasy psa na podstawie zdjęcia.
- Obsługa przesyłania zdjęć w formacie JPEG/PNG.
- Opcjonalna obsługa bounding boxów.

---

## Jak uruchomić kontener

1. **Zbuduj obraz Dockera:**

   W katalogu z plikiem `Dockerfile` uruchom:
   ```bash
   docker build -t dog-breed-api .

2. **Uruchom kontener:**

   Uruchom serwer API w kontenerze na porcie 8000:
   ```bash
   docker run -p 8000:8000 dog-breed-api

3. **Sprawdź, czy API działa:**

   Otwórz przeglądarkę i przejdź do adresu:
   [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

   Powinieneś zobaczyć interaktywną dokumentację Swagger wygenerowaną przez FastAPI.

---

## Jak uruchomić serwis REST API

1. **Uruchom kontener według instrukcji powyżej.**
2. **Dostęp do API:**
   - **Swagger UI:** Dokumentacja API z możliwością testowania żądań znajduje się pod adresem:
     [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - **Endpoint do przewidywań:** Endpoint REST API znajduje się pod adresem:
     ```
     POST http://127.0.0.1:8000/predict/
     ```

---

## Jak testować przewidywania

### 1. Przy użyciu `curl`

Możesz przesłać zdjęcie do API przy użyciu narzędzia `curl`. Przykładowe zapytanie:


```bash
curl -X POST "http://127.0.0.1:8000/predict/" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-F "file=@path_to_your_image.jpg" \
-F "xmin=10" \
-F "ymin=20" \
-F "xmax=150" \
-F "ymax=200"
```

### 2. Przy użyciu Postman

1. Otwórz aplikację Postman.
2. Utwórz nowe zapytanie:
   - **Metoda:** POST
   - **URL:** `http://127.0.0.1:8000/predict/`
3. W sekcji **Body** wybierz **form-data** i dodaj:
   - Pole `file` (Typ: **File**) → Załaduj obraz z dysku.
   - Pole `xmin` → Wpisz wartość 10.
   - Pole `ymin` → Wpisz wartość 20.
   - Pole `xmax` → Wpisz wartość 150.
   - Pole `ymax` → Wpisz wartość 200.
4. Kliknij **Send**, aby wysłać zapytanie.

**Przykładowa odpowiedź w Postman:**
```json
{
  "predicted_label": "bloodhound"
}
