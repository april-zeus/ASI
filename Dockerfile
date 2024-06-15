# Użycie obrazu Python 3.10
FROM python:3.10
LABEL authors="s22488"

# Ustawienie katalogu roboczego
WORKDIR /app

# Instalacja niezbędnych pakietów systemowych
RUN apt-get -y update && apt-get install -y \
    python3-dev \
    apt-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Aktualizacja setuptools
RUN pip3 install --upgrade setuptools

# Skopiowanie pliku requirements.txt i instalacja zależności
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Skopiowanie reszty kodu aplikacji
COPY . .

# Utworzenie katalogów dla modeli i danych
RUN mkdir -p /app/ml_models /app/data

# Ustawienie zmiennej środowiskowej dla portu
EXPOSE 8000
EXPOSE 8501

# Komenda uruchamiająca aplikację
CMD python launcher.py script
