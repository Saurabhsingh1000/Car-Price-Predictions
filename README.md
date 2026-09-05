# Car Price Prediction

An end-to-end machine learning system that estimates the fair resale price of a used car in the Indian market, served through a REST API and a small web UI.

This is a from-scratch rebuild of an earlier notebook-only version of this project (a single Linear Regression model + a Flask app with hardcoded local file paths). It's now a reproducible pipeline with proper train/test methodology, model comparison, a tested FastAPI backend, Docker packaging, and CI.

## Overview

Give the model a car's brand, model name, manufacturing year, kilometers driven, and fuel type, and it returns an estimated selling price in INR. The project demonstrates the full lifecycle of a small, honest ML product: messy raw data → cleaning → feature engineering → model comparison → hyperparameter tuning → a served, tested API.

## Problem Statement

Used-car sellers and buyers in India often rely on gut feeling or a few manually-checked listings to judge whether a price is fair. This project trains a regression model on real historical listing data to produce a quick, data-driven price estimate — a starting point for negotiation, not a formal valuation.

## Dataset

The raw dataset (`data/raw/car_dataset.csv`) is scraped resale-listing data with **892 rows** and 6 columns: `name`, `company`, `year`, `Price`, `kms_driven`, `fuel_type`. It is messy in realistic ways:

- `year` sometimes contains non-numeric placeholders.
- `Price` is a string, sometimes literally `"Ask For Price"`, using Indian comma grouping (e.g. `"4,25,000"`).
- `kms_driven` is a string like `"45,000 kms"`.
- `fuel_type` has missing values.
- `name` is a long, inconsistent free-text listing title.

There is **no transmission, engine capacity, mileage, seats, or owner-type column** in this dataset — earlier drafts of this project's brief assumed those fields exist, but they don't, so the pipeline and UI only use the columns that are actually present: `name`, `company`, `year`, `kms_driven`, `fuel_type`, plus two engineered features. No columns were invented.

After cleaning (`src/data/preprocessing.py`), **723 rows** remain, covering **25 brands**, **254 distinct model names**, model years **1995–2019**, and prices from ₹30,000 to just under ₹60,00,000.

## Features

| Feature | Type | Source |
|---|---|---|
| `name` | categorical | raw (truncated to first 3 words) |
| `company` | categorical | raw |
| `year` | numeric | raw |
| `kms_driven` | numeric | raw |
| `fuel_type` | categorical (Petrol / Diesel / LPG) | raw |
| `car_age` | numeric, engineered | `2024 - year` |
| `kms_per_year` | numeric, engineered | `kms_driven / max(car_age, 1)` |

## Machine Learning Approach

A scikit-learn `Pipeline` wraps a `ColumnTransformer` (one-hot encoding for categoricals, standard scaling for numerics) and a regressor, so the exact preprocessing used in training is guaranteed at inference time — there is no separate "prediction-time" cleaning logic to drift out of sync.

## Data Preprocessing

See `src/data/preprocessing.py`. In order: drop duplicates → keep numeric `year` → drop `"Ask For Price"` rows → parse `Price`/`kms_driven` into integers → drop missing `fuel_type` → truncate `name` to 3 words → drop price outliers (`< ₹10,000` or `≥ ₹60,00,000`, matching the original EDA's outlier boundary) → drop unrealistic kms values → reset index.

## Feature Engineering

`src/features/engineering.py` adds:
- **`car_age`** — buyers reason about "how old" more naturally than an absolute year, and it lets the model generalize slightly better to unseen years.
- **`kms_per_year`** — usage intensity. A 2-year-old car with 40,000 km has been driven much harder than a 15-year-old one with the same total, and this feature captures that.

## Models Compared

Four baseline regressors were compared with 5-fold cross-validation (scoring: negative MAE) on the training split only, before the test set was ever touched:

| Model | CV MAE (₹) | CV Std (₹) |
|---|--:|--:|
| **Ridge Regression** | **123,099** | 14,482 |
| Random Forest | 126,524 | 15,941 |
| Gradient Boosting | 139,573 | 17,250 |
| Hist Gradient Boosting | 194,446 | 22,999 |

Ridge came out on top. This is a real (if slightly counter-intuitive) result of the dataset's shape: with only 578 training rows and a high-cardinality `name` column (254 unique model names, one-hot encoded), tree-based ensembles have too little data per split to beat a well-regularized linear model. This kind of "the simple model wins" outcome is common with small, sparse, high-cardinality tabular data, and it's worth understanding rather than papering over with a fancier model.

`RandomizedSearchCV` (20 iterations, 5-fold CV) then tuned Ridge's `alpha`, improving CV MAE to **₹116,003** at `alpha=0.1`.

## Model Evaluation

Final metrics on the **held-out test set** (145 rows, never used in training or model selection):

| Model | MAE | RMSE | R² |
|---|--:|--:|--:|
| Ridge (tuned, alpha=0.1) | ₹129,824 | ₹281,525 | 0.532 |

(MAPE: 45.6% — inflated by a handful of very cheap, older cars where even a small absolute error is a large percentage error; MAE/RMSE in rupees are the more informative numbers here.)

These are the actual numbers produced by `scripts/train_model.py` and reproduced independently by `scripts/evaluate_model.py` — see `models/metrics.json` after training.

## Best Model

**Ridge Regression (alpha=0.1)** was selected because it had the lowest cross-validation MAE both before and after tuning. It's also the most interpretable option here, and its small training cost (a fraction of a second) makes retraining trivial as new listing data comes in.

## Project Architecture

```
car-price-prediction/
│
├── data/
│   ├── raw/car_dataset.csv
│   └── processed/cleaned_car.csv        # generated by the pipeline
│
├── notebooks/
│   └── 01_eda_and_model_experiments.ipynb
│
├── src/
│   ├── data/preprocessing.py
│   ├── features/engineering.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── utils/
│       ├── config.py
│       └── logging_config.py
│
├── app/
│   ├── main.py                # FastAPI app
│   ├── schemas.py              # Pydantic request/response models
│   ├── templates/index.html
│   └── static/{style.css, app.js}
│
├── models/                     # trained pipeline + metadata (generated)
├── tests/
│   ├── test_preprocessing.py
│   ├── test_prediction.py
│   └── test_api.py
├── scripts/
│   ├── train_model.py
│   └── evaluate_model.py
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── .env.example
└── README.md
```

## API

Interactive OpenAPI docs are auto-generated at `/docs` once the server is running.

**`GET /health`**
```json
{ "status": "ok", "model_loaded": true }
```

**`POST /predict`**

Request:
```json
{
  "name": "Hyundai Grand i10",
  "company": "Hyundai",
  "year": 2016,
  "kms_driven": 30000,
  "fuel_type": "Petrol"
}
```

Response:
```json
{ "predicted_price": 382025.98, "currency": "INR" }
```

Invalid input (e.g. `fuel_type: "Electric"`, or `year` out of range) returns a `422` with a readable `detail` message — never a raw stack trace. A missing model artifact returns `503`.

**`GET /meta`** — known brands/fuel types/year range, used to populate the UI's dropdowns.

## Running Locally

```bash
git clone <your-fork-url>
cd car-price-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/train_model.py     # trains the model, writes models/*.joblib + metrics
uvicorn app.main:app --reload     # serves the API + UI at http://localhost:8000
```

Then open `http://localhost:8000` for the web UI, or `http://localhost:8000/docs` for the API docs.

## Running with Docker

```bash
docker build -t car-price-prediction .
docker run --rm -p 8000:8000 car-price-prediction
```

or with Compose:
```bash
docker-compose up --build
```

The image trains the model at build time, so the container is self-contained — no separate training step or mounted volume is required to get a working API.

## Testing

```bash
pip install -r requirements-dev.txt
python scripts/train_model.py   # a few tests need a trained model artifact present
pytest tests/ -v
```

24 tests cover data cleaning behaviour (not just imports), feature engineering, end-to-end prediction (including unseen-category handling), and the API (health, valid/invalid prediction payloads, and a check that a simulated internal error never leaks its message to the client).

## Screenshots

> _[Add a screenshot of the web UI here, e.g. `docs/screenshot-ui.png`]_
> _[Add a screenshot of the `/docs` Swagger UI here]_

## Future Improvements

- Add more informative raw features if a richer dataset becomes available (transmission, engine size, owner count, seats) — deliberately not faked here.
- Track experiments with MLflow instead of flat JSON files.
- Add a simple drift-monitoring job comparing incoming prediction inputs to the training distribution.
- Deploy the container to a cloud provider (not done yet — this repo runs locally/Docker only).
- Explore target-encoding `name`/`company` instead of one-hot, which would likely help the tree-based models close the gap with Ridge.

## Skills Demonstrated

Python · pandas/NumPy data cleaning · feature engineering · scikit-learn Pipelines & ColumnTransformer · cross-validation & hyperparameter tuning (RandomizedSearchCV) · regression model comparison and evaluation (MAE/RMSE/R²/MAPE) · FastAPI + Pydantic validation · automated testing with pytest · Docker · GitHub Actions CI · reproducible, config-driven project structure.

## Limitations

- The dataset is small (723 clean rows) and several years out of date (listings up to 2019), so absolute price predictions should be treated as directional, not authoritative.
- No transmission, engine, or condition data — some obviously price-relevant information simply isn't in this dataset.
- Test R² of 0.53 and MAPE of ~46% mean the model explains roughly half the price variance; it's a solid baseline demonstrating the full pipeline, not a production valuation tool.
- No cloud deployment is included in this repository.
