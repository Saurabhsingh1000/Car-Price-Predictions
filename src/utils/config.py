"""
Central configuration for the Car Price Prediction project.

All paths are computed relative to the project root so the code works the
same way regardless of the current working directory it is invoked from.
"""

from __future__ import annotations

from pathlib import Path

# Project root = two levels up from this file (src/utils/config.py -> project root)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "car_dataset.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaned_car.csv"

# ---------------------------------------------------------------------------
# Model artifact paths
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "car_price_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
CURRENT_YEAR = 2024  # dataset was scraped ~2019-2020; kept fixed for reproducibility
MAX_REASONABLE_PRICE = 6_000_000  # INR, matches an outlier boundary found during EDA
MIN_REASONABLE_PRICE = 10_000  # INR

CATEGORICAL_FEATURES = ["name", "company", "fuel_type"]
NUMERICAL_FEATURES = ["year", "kms_driven", "car_age", "kms_per_year"]

TARGET_COLUMN = "price"

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
CURRENCY = "INR"
