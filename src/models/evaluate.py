"""
Standalone evaluation script: loads the persisted pipeline and re-computes
test-set metrics. Useful for CI or for verifying a saved model without
re-running the full training + hyperparameter search.

Since the training split is deterministic (fixed ``RANDOM_SEED`` and
``TEST_SIZE`` from ``src.utils.config``), re-splitting the cleaned data here
reproduces the exact same held-out test set that was used during training.
"""

from __future__ import annotations

import json

import joblib
from sklearn.model_selection import train_test_split

from src.data.preprocessing import load_and_clean
from src.features.engineering import add_engineered_features, split_features_target
from src.models.train import evaluate_on_test
from src.utils.config import (
    MODEL_PATH,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def evaluate_saved_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run `python scripts/train_model.py` first."
        )

    df = load_and_clean(save=False)
    df = add_engineered_features(df)
    X, y = split_features_target(df, TARGET_COLUMN)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    pipeline = joblib.load(MODEL_PATH)
    metrics = evaluate_on_test(pipeline, X_test, y_test)
    logger.info("Evaluation complete: %s", json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    evaluate_saved_model()
