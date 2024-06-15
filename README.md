# Projekt z przedmiotu “Architektury rozwiązań i wdrożeń SI”

Streamlit app will be open in browser on URL http://localhost:8501/, and you will be able to manage software from this web panel.

## Churn modelling

Autorzy:
* Konrad Reperowski
* Barnaba Gańko
* Adam Kwiecień

Projekt służy do tworzenia modeli do predykcji rezygnacji z usługi konta bankowego na podstawie danych dostępnych w serwisie [Kaggle](https://www.kaggle.com/code/simgeerek/churn-prediction-using-machine-learning/input). 


Projekt korzysta z:
* SQLite - baza danych
* Kedro - zarządzanie danymi oraz pipeline'ami
* Kedro Viz - wizualizacja pipeline'ów
* AutoGluon - dostosowanie modeli
* WandDB - śledzenie i wizualizacja treningu modeli
* SDV - generator syntetycznych danych
* FastAPI - tworzenie API i interakcji z modelem
* Streamlit - prosty interfejs użytkownika

### Aktywacja środowiska wirtualnego: 
```bash
source .venv/bin/activate
```

### Dezaktywacja środowiska wirtualnego: 
```bash
deactivate
```

### Baza danych

* CREDENTIALS=db-credentials
* TABLE_NAME=raw_data
* TRAIN_TABLE_NAME=train_data
* TEST_TABLE_NAME=test_data
* EVALUATION_TABLE_NAME=evaluation_metrics
* CONFUSION_MATRIX_TABLE_NAME=confusion_matrix
* SYNTH_TABLE_NAME=synth_data

### Dane:

* RowNumber: odpowiada numerowi rekordu (wiersza) i nie ma wpływu na wynik.
* CustomerId: zawiera losowe wartości i nie ma wpływu na to, czy klient odejdzie z banku.
* CreditScore: może mieć wpływ na odejście klienta, ponieważ klient z wyższą oceną kredytową jest mniej skłonny do opuszczenia banku.
* Geography: lokalizacja klienta może wpływać na jego decyzję o odejściu z banku.
* Gender: interesujące jest zbadanie, czy płeć ma wpływ na odejście klienta z banku.
* Age: jest to z pewnością istotne, ponieważ starsi klienci są mniej skłonni do opuszczenia banku niż młodsi.
* Tenure: odnosi się do liczby lat, przez które klient jest klientem banku. Zazwyczaj starsi klienci są bardziej lojalni i mniej skłonni do opuszczenia banku.
* Balance: również jest bardzo dobrym wskaźnikiem odejścia klienta, ponieważ osoby z wyższym saldem na kontach są mniej skłonne do opuszczenia banku w porównaniu z tymi, którzy mają niższe salda.
* NumOfProducts: odnosi się do liczby produktów, które klient zakupił za pośrednictwem banku.
* HasCrCard: oznacza, czy klient posiada kartę kredytową. Ta kolumna jest również istotna, ponieważ osoby z kartą kredytową są mniej skłonne do opuszczenia banku.
* IsActiveMember: aktywni klienci są mniej skłonni do opuszczenia banku.
* EstimatedSalary: podobnie jak w przypadku salda, osoby o niższych dochodach są bardziej skłonne do opuszczenia banku w porównaniu z tymi, którzy mają wyższe wynagrodzenia.
* Exited: czy klient opuścił bank. (0 = Nie, 1 = Tak)

## Instalacja i uruchomienie

```bash
pip install -r requirements.txt
```

```bash
python launcher.py script
```

## Wizualizacja pipeline'ów

```bash
cd churn-modelling-kedro
kedro viz
```

![kedro-pipeline.png](kedro-pipeline.png)

## Architektura projektu

![architecture.png](architecture.png)