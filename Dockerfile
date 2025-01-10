# Bazowy obraz
FROM python:3.9-slim

# Ustawienie katalogu roboczego
WORKDIR /app

# Kopiowanie plików aplikacji
COPY . /app

# Instalowanie zależności
RUN pip install --no-cache-dir fastapi uvicorn tensorflow numpy pydantic pillow scikit-learn python-multipart

# Eksponowanie portu
EXPOSE 5000

# Uruchomienie serwisu FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
