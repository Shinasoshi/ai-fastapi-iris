# AI Web Service -- Iris Classification API

## 1. Opis systemu

Projekt przedstawia prosty system sztucznej inteligencji udostępniony
jako usługa webowa (API).

System: - wykorzystuje wcześniej wytrenowany model uczenia maszynowego
(scikit-learn), - udostępnia model poprzez aplikację webową w FastAPI, -
przyjmuje dane wejściowe w formacie JSON, - zwraca wynik predykcji jako
odpowiedź HTTP.

Model rozwiązuje problem klasyfikacji zbioru Iris -- na podstawie
czterech cech kwiatu przewiduje jego gatunek.

Projekt łączy zagadnienia: - uczenia maszynowego (Python +
scikit-learn), - tworzenia aplikacji webowych (FastAPI), - zarządzania
środowiskiem (`uv`), - pracy zespołowej z wykorzystaniem Git.

------------------------------------------------------------------------

# 2. Instalacja i uruchomienie

## Wymagania

-   Python 3.11 lub nowszy
-   narzędzie `uv`
-   Git

------------------------------------------------------------------------

## Przygotowanie środowiska (`uv`)

Po sklonowaniu repozytorium przejdź do katalogu projektu:

``` bash
git clone https://github.com/Shinasoshi/ai-fastapi-iris.git
cd ai-fastapi-iris
```

Zainstaluj zależności:

``` bash
uv sync --extra dev
```

------------------------------------------------------------------------

## Trenowanie modelu

Model jest trenowany lokalnie i zapisywany do pliku.\
Nie jest trenowany przy każdym uruchomieniu serwera.

Aby wytrenować model:

``` bash
uv run python scripts/train_model.py
```

Model zostanie zapisany w:

    models/iris_model.joblib

------------------------------------------------------------------------

## Uruchomienie serwera FastAPI

``` bash
uv run uvicorn app.main:app --reload
```

Serwer będzie dostępny pod adresem:

    http://127.0.0.1:8000

Automatyczna dokumentacja API (Swagger UI):

    http://127.0.0.1:8000/docs

------------------------------------------------------------------------

# 3. Instrukcja użycia API

## Endpoint: GET /health

Sprawdza, czy model został poprawnie załadowany.

### Przykładowa odpowiedź:

``` json
{
  "status": "ok",
  "model_loaded": true,
  "error": null
}
```

------------------------------------------------------------------------

## Endpoint: POST /predict

Wykonuje predykcję gatunku kwiatu Iris.

### Dane wejściowe (JSON)

``` json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Dane wyjściowe (JSON)

``` json
{
  "predicted_class": 0,
  "predicted_label": "setosa",
  "probabilities": [0.98, 0.01, 0.01]
}
```

------------------------------------------------------------------------

# 4. Informacje o modelu

-   Typ problemu: klasyfikacja
-   Biblioteka: scikit-learn
-   Model: Pipeline (StandardScaler + LogisticRegression)
-   Dane: zbiór Iris (wbudowany w scikit-learn)
-   Liczba klas: 3
-   Metryka ewaluacyjna: accuracy (obliczana podczas treningu)

------------------------------------------------------------------------

# 5. Dane wejściowe i wyjściowe

## Dane wejściowe:

Cztery wartości liczbowe (float): - sepal_length -- długość działki
kielicha - sepal_width -- szerokość działki kielicha - petal_length --
długość płatka - petal_width -- szerokość płatka

## Dane wyjściowe:

-   predicted_class -- numer klasy (0--2)
-   predicted_label -- nazwa gatunku
-   probabilities -- lista prawdopodobieństw dla każdej klasy

------------------------------------------------------------------------

# 6. Testowanie

Aby uruchomić testy:

``` bash
uv run pytest
```

------------------------------------------------------------------------

# 7. Praca zespołowa

Projekt realizowany w dwuosobowym zespole z wykorzystaniem systemu
kontroli wersji Git.

Współpraca obejmowała: - podział zadań (model ML / warstwa API), -
wspólne repozytorium, - pracę na branchach, - integrację zmian, -
przygotowanie wspólnej dokumentacji.
