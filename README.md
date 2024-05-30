# Projekt z przedmiotu “Architektury rozwiązań i wdrożeń SI”

## Churn modelling

Autorzy:
* Konrad Reperowski
* Barnaba Gańko
* Adam Kwiecień

[TODO: Dokumentacja techniczna »](./docs/TODO-dokumentacja-techniczna.pdf)

Projekt służy do tworzenia modeli do predykcji rezygnacji z usługi konta bankowego na podstawie danych dostępnych w serwisie [Kaggle](https://www.kaggle.com/code/simgeerek/churn-prediction-using-machine-learning/input). 


Projekt korzysta z:
* SQLite - baza danych
* Kedro - zarządzanie danymi oraz pipeline'ami
* Kedro Viz - wizualizacja pipeline'ów
* AutoGluon - dostosowanie modeli
* WandDB - śledzenie i wizualizacja treningu modeli
* FastAPI - tworzenie API i interakcji z modelem
* Streamlit - prosty interfejs użytkownika

## Development (TODO)

Aktywacja środowiska wirtualnego: 
```bash
source .venv/bin/activate
```

Dezaktywacja środowiska wirtualnego: 
```bash
deactivate
```

[TO REMOVE] Kamienie milowe
* Stworzenie repozytorium ✅
* Konfiguracja środowiska wirtualnego ✅
* Instalacja Kedro i Kedro Viz ✅
* Instalacja pozostałych paczek ML-owych ⚙️
* Konfiguracja pipeline'ów ⚙️
  * data_engineering pipeline ✅
  * data_science pipeline ✅
  * model_evaluation pipeline ⚙️
  * model_retraining pipeline ⚙️
  * synthetic_data_creation pipeline ⚙️

[TO REMOVE] Pomocne linki:
* [Kedro tutorial](https://neptune.ai/blog/data-science-pipelines-with-kedro)
* [Sample project](https://github.com/KarolChlasta/ASI/blob/main/7_project_examples/Beta/asi-kedro/conf/base/catalog.yml)

### Baza danych

* CREDENTIALS=db-credentials
* TABLE_NAME=raw_data
* TRAIN_TABLE_NAME=train_data
* TEST_TABLE_NAME=test_data
* EVALUATION_TABLE_NAME=evaluation_metrics
* CONFUSION_MATRIX_TABLE_NAME=confusion_matrix
* SYNTH_TABLE_NAME=synth_data

## Instalacja i uruchomienie (TODO)