"""Tests for feature engineering and the end-to-end prediction function.

These tests require a trained model artifact to exist at
``models/car_price_pipeline.joblib`` (produced by ``scripts/train_model.py``).
"""

import pandas as pd
import pytest

from src.features.engineering import add_engineered_features, split_features_target
from src.models.predict import load_pipeline, predict_price
from src.utils.config import MODEL_PATH

MODEL_AVAILABLE = MODEL_PATH.exists()
NO_MODEL_REASON = "Trained model artifact not found; run scripts/train_model.py first."


def test_add_engineered_features_computes_car_age():
    df = pd.DataFrame({"year": [2020, 2010], "kms_driven": [10000, 50000]})
    result = add_engineered_features(df)
    assert "car_age" in result.columns
    assert "kms_per_year" in result.columns
    assert (result["car_age"] >= 0).all()


def test_add_engineered_features_handles_current_year_car_without_div_by_zero():
    df = pd.DataFrame({"year": [2024], "kms_driven": [500]})
    result = add_engineered_features(df)
    assert result["car_age"].iloc[0] == 0
    # Should not raise ZeroDivisionError / produce inf
    assert result["kms_per_year"].iloc[0] == 500


def test_split_features_target_separates_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "price": [100, 200]})
    X, y = split_features_target(df, "price")
    assert "price" not in X.columns
    assert list(y) == [100, 200]


@pytest.mark.skipif(not MODEL_AVAILABLE, reason=NO_MODEL_REASON)
def test_load_pipeline_returns_fitted_estimator():
    pipeline = load_pipeline()
    assert hasattr(pipeline, "predict")


@pytest.mark.skipif(not MODEL_AVAILABLE, reason=NO_MODEL_REASON)
def test_predict_price_returns_positive_float():
    price = predict_price(
        name="Hyundai Grand i10",
        company="Hyundai",
        year=2016,
        kms_driven=30000,
        fuel_type="Petrol",
    )
    assert isinstance(price, float)
    assert price > 0


@pytest.mark.skipif(not MODEL_AVAILABLE, reason=NO_MODEL_REASON)
def test_predict_price_handles_unseen_categories_gracefully():
    # An unknown company/name should not raise — the OneHotEncoder is
    # configured with handle_unknown="ignore".
    price = predict_price(
        name="Totally Unknown Car",
        company="UnknownBrand",
        year=2019,
        kms_driven=15000,
        fuel_type="Diesel",
    )
    assert isinstance(price, float)


@pytest.mark.skipif(not MODEL_AVAILABLE, reason=NO_MODEL_REASON)
def test_predict_price_older_higher_mileage_car_is_cheaper_on_average():
    newer_cheaper_use = predict_price(
        name="Maruti Suzuki Swift",
        company="Maruti",
        year=2019,
        kms_driven=10000,
        fuel_type="Petrol",
    )
    older_higher_use = predict_price(
        name="Maruti Suzuki Swift",
        company="Maruti",
        year=2008,
        kms_driven=120000,
        fuel_type="Petrol",
    )
    assert newer_cheaper_use > older_higher_use
