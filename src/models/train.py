"""
Model training pipeline for used-car price prediction.

Workflow:
  1. Load + clean raw data, engineer features.
  2. Train/test split (test set is held out untouched until final evaluation).
  3. Build a preprocessing ColumnTransformer (one-hot encode categoricals,
     scale numeric features) wrapped in a scikit-learn Pipeline so the exact
     same transformation is guaranteed at inference time.
  4. Train several baseline regressors and compare them with 5-fold CV on
     the training set (never touching the test set) using negative MAE.
  5. Hyperparameter-tune the best-performing model family with
     RandomizedSearchCV.
  6. Refit the tuned pipeline on the full training set and evaluate once on
     the held-out test set.
  7. Persist the final pipeline (joblib), metadata (JSON) and metrics (JSON).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.preprocessing import load_and_clean
from src.features.engineering import add_engineered_features, split_features_target
from src.utils.config import (
    CATEGORICAL_FEATURES,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    NUMERICAL_FEATURES,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    best_model_name: str
    cv_scores: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    best_params: dict = field(default_factory=dict)


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer used for both training and inference."""
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numerical_transformer = StandardScaler()

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numerical_transformer, NUMERICAL_FEATURES),
        ]
    )


def get_candidate_models() -> dict:
    """Baseline model zoo to compare via cross-validation."""
    return {
        "ridge": Ridge(random_state=RANDOM_SEED),
        "random_forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_SEED),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            random_state=RANDOM_SEED
        ),
    }


def get_search_space(model_name: str) -> dict:
    """Hyperparameter search spaces for the models eligible for tuning."""
    spaces = {
        "random_forest": {
            "model__n_estimators": [100, 200, 300, 400],
            "model__max_depth": [None, 8, 12, 16, 24],
            "model__min_samples_split": [2, 4, 6, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        },
        "gradient_boosting": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 4, 5],
            "model__subsample": [0.7, 0.85, 1.0],
        },
        "hist_gradient_boosting": {
            "model__max_iter": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [None, 4, 6, 10],
            "model__max_leaf_nodes": [15, 31, 63],
        },
        "ridge": {
            "model__alpha": [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        },
    }
    return spaces[model_name]


def compare_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Run 5-fold CV for every candidate model and return MAE scores."""
    preprocessor = build_preprocessor()
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    results = {}

    for name, model in get_candidate_models().items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        start = time.time()
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        mae_scores = -scores
        elapsed = time.time() - start
        results[name] = {
            "cv_mae_mean": float(mae_scores.mean()),
            "cv_mae_std": float(mae_scores.std()),
            "seconds": round(elapsed, 2),
        }
        logger.info(
            "Model=%-22s CV MAE = %.0f (+/- %.0f) [%.1fs]",
            name,
            mae_scores.mean(),
            mae_scores.std(),
            elapsed,
        )

    return results


def tune_best_model(
    best_model_name: str, X_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    """Run RandomizedSearchCV for the best candidate model."""
    preprocessor = build_preprocessor()
    model = get_candidate_models()[best_model_name]
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    param_distributions = get_search_space(best_model_name)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        refit=True,
    )
    logger.info("Starting hyperparameter search for %s (20 iterations, 5-fold CV)", best_model_name)
    search.fit(X_train, y_train)
    logger.info("Best CV MAE after tuning: %.0f", -search.best_score_)
    logger.info("Best params: %s", search.best_params_)
    return search.best_estimator_, search.best_params_, -search.best_score_


def evaluate_on_test(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute regression metrics on the untouched test set."""
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape = float(np.mean(np.abs((y_test - preds) / y_test)) * 100)

    metrics = {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4),
        "mape": round(mape, 2),
        "n_test_samples": int(len(y_test)),
    }
    logger.info("Test metrics: %s", metrics)
    return metrics


def run_training_pipeline() -> TrainingResult:
    """Full end-to-end training pipeline. Returns a TrainingResult summary."""
    # 1. Load + clean + engineer
    df = load_and_clean(save=True)
    df = add_engineered_features(df)
    X, y = split_features_target(df, TARGET_COLUMN)

    # 2. Train / test split (test set held out entirely until the very end)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    logger.info("Train size=%d, Test size=%d", len(X_train), len(X_test))

    # 3-4. Compare baseline models via cross-validation on the training set
    cv_results = compare_models(X_train, y_train)
    best_model_name = min(cv_results, key=lambda k: cv_results[k]["cv_mae_mean"])
    logger.info("Best baseline model by CV MAE: %s", best_model_name)

    # 5. Hyperparameter tuning for the winning model family
    best_pipeline, best_params, tuned_cv_mae = tune_best_model(
        best_model_name, X_train, y_train
    )

    # 6. Final evaluation on the untouched test set
    test_metrics = evaluate_on_test(best_pipeline, X_test, y_test)

    # 7. Persist artifacts
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    logger.info("Saved trained pipeline to %s", MODEL_PATH)

    metadata = {
        "best_model_name": best_model_name,
        "best_params": best_params,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "target": TARGET_COLUMN,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "known_companies": sorted(df["company"].unique().tolist()),
        "known_names": sorted(df["name"].unique().tolist()),
        "known_fuel_types": sorted(df["fuel_type"].unique().tolist()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model metadata to %s", METADATA_PATH)

    full_metrics = {
        "cross_validation": cv_results,
        "tuned_cv_mae": round(float(tuned_cv_mae), 2),
        "test_set": test_metrics,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(full_metrics, f, indent=2)
    logger.info("Saved metrics to %s", METRICS_PATH)

    return TrainingResult(
        best_model_name=best_model_name,
        cv_scores=cv_results,
        test_metrics=test_metrics,
        best_params=best_params,
    )


if __name__ == "__main__":
    run_training_pipeline()
