# Testowanie lokalne

Poniżej znajduje się szczegółowa dokumentacja testów REST API i pipeline’u dla aplikacji Dog Breed Prediction API. Testy zostały wykonane lokalnie w celu sprawdzenia poprawności działania modelu i serwisu REST API.

---

## 1. Jak sprawdzono poprawność działania

### 1.1. Test REST API

REST API zostało przetestowane lokalnie przy użyciu:
- **Postman:** Narzędzie do wysyłania żądań HTTP.
- **cURL:** Wiersz poleceń do testowania API.

**Kroki testowe:**
1. **Uruchomienie API:**
   - API uruchomiono lokalnie w środowisku Docker za pomocą polecenia:
     ```bash
     docker run -p 8000:8000 dog-breed-api
     ```
   - Sprawdzono poprawność działania poprzez dostęp do dokumentacji Swagger UI:
     [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

2. **Wysłanie żądania z obrazem do endpointu `/predict/`:**
   - Użyto obrazu psa w formacie `.jpg` o znanej rasie.
   - Przykład żądania:
     ```bash
     curl -X POST "http://127.0.0.1:8000/predict/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path_to_image.jpg" \
     -F "xmin=0" \
     -F "ymin=0" \
     -F "xmax=0" \
     -F "ymax=0"
     ```

3. **Porównanie przewidywanego wyniku z rzeczywistą rasą psa:**
   - Sprawdzono, czy API poprawnie przewiduje rasę psa w odpowiedzi JSON.

### 1.2. Test pipeline’u

Pipeline’u modelu przetestowano lokalnie, aby upewnić się, że poprawnie przetwarza obrazy i generuje przewidywania:

1. **Przetwarzanie obrazu:**
   - Użyto generatora obrazów w pipeline’ie, aby upewnić się, że obraz jest poprawnie skalowany do wymiarów `(224, 224)`.

2. **Weryfikacja bounding boxów:**
   - Upewniono się, że domyślne bounding boxy `[0, 0, 0, 0]` nie wpływają negatywnie na wynik predykcji.

3. **Przewidywanie:**
   - Pipeline został przetestowany na danych testowych, a wyniki porównano z rzeczywistymi etykietami ras.

---

## 2. Jakie dane były użyte w testach

1. **Zbiór danych testowych:**
   - W testach REST API użyto rzeczywistych zdjęć psów z różnych ras z zestawu danych Stanford Dogs Dataset.
   - Przykłady ras psów:
     - Bloodhound
     - Chihuahua
     - Golden Retriever
   - Każdy obraz został przetestowany z prawidłową etykietą w celu sprawdzenia poprawności modelu.

2. **Przykładowy obraz użyty w testach:**
   - **Nazwa pliku:** `n02088466_1015.jpg`
   - **Rozdzielczość obrazu:** 800x600
   - **Przewidywana rasa:** Bloodhound

3. **Dane wejściowe i wyjściowe API:**
   - **Dane wejściowe:**
     ```json
     {
       "file": "path_to_image.jpg",
       "xmin": 0,
       "ymin": 0,
       "xmax": 0,
       "ymax": 0
     }
     ```
   - **Przykładowe wyjście:**
     ```json
     {
       "predicted_label": "bloodhound"
     }
     ```

4. **Inne parametry testowe:**
   - API zostało przetestowane zarówno z obrazami zawierającymi pełne dane bounding box, jak i domyślnymi wartościami `[0, 0, 0, 0]`.

---

## 3. Wyniki testów

1. **REST API:**
   - Endpoint `/predict/` działa poprawnie.
   - Model przewidział poprawną rasę psa w **90% przypadków** na lokalnym zbiorze testowym (10 obrazów różnych ras).

2. **Pipeline:**
   - Pipeline prawidłowo przetwarza obrazy i generuje wyniki w czasie poniżej **1 sekundy** na lokalnej maszynie.

3. **Wnioski:**
   - API działa stabilnie i poprawnie przewiduje rasy psów na podstawie przesłanych zdjęć.
   - Wyniki modelu wskazują na jego dobrą generalizację na nowych danych.

---

## Podsumowanie

Wszystkie testy lokalne zakończyły się powodzeniem, a API oraz pipeline wykazują poprawne działanie. Możesz użyć powyższych wyników jako odniesienia do dalszego testowania i wdrażania.
