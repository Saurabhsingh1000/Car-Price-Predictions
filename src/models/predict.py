"""
Inference utilities: load the persisted pipeline once and expose a simple
``predict_price`` function used by both the API and the tests.

The pipeline saved by ``src.models.train`` already contains the fitted
ColumnTransformer, so raw (but feature-engineered) input rows can be passed
straight through with no risk of train/serve preprocessing skew.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.features.engineering import add_engineered_features
from src.utils.config import METADATA_PATH, MODEL_PATH
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelNotFoundError(RuntimeError):
    """Raised when the trained pipeline artifact cannot be found on disk."""


@lru_cache(maxsize=1)
def load_pipeline(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        raise ModelNotFoundError(
            f"No trained model found at {model_path}. "
            "Run `python scripts/train_model.py` to train and save one."
        )
    logger.info("Loading trained pipeline from %s", model_path)
    return joblib.load(model_path)


@lru_cache(maxsize=1)
def load_metadata(metadata_path: Path = METADATA_PATH) -> dict:
    if not metadata_path.exists():
        return {}
    with open(metadata_path) as f:
        return json.load(f)


def predict_price(
    name: str,
    company: str,
    year: int,
    kms_driven: int,
    fuel_type: str,
) -> float:
    """Predict the selling price (in the dataset's currency unit, INR) for
    a single car described by its raw attributes.

    The same ``add_engineered_features`` used at training time is applied
    here so the model sees an identically-shaped feature row.
    """
    pipeline = load_pipeline()

    row = pd.DataFrame(
        [
            {
                "name": name,
                "company": company,
                "year": year,
                "kms_driven": kms_driven,
                "fuel_type": fuel_type,
            }
        ]
    )
    row = add_engineered_features(row)

    prediction = pipeline.predict(row)[0]
    return round(float(prediction), 2)
