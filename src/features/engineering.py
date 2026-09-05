"""
Feature engineering for the used-car price model.

Two derived features are added on top of the cleaned raw columns:

* ``car_age``      = reference year - manufacturing year. Buyers reason about
                      "how old is this car" more naturally than an absolute
                      year, and it also lets the model extrapolate slightly
                      better to years not seen during training.
* ``kms_per_year``  = kms_driven / max(car_age, 1). Two cars with the same
                      total kms_driven can be very different (a 2-year-old
                      car with 40,000 km has been driven much harder than a
                      15-year-old one with 40,000 km). This captures usage
                      intensity, which is informative for depreciation.

All engineering here is pure/deterministic and is applied identically at
training and inference time via ``add_engineered_features`` so there is no
train/serve skew.
"""

from __future__ import annotations

import pandas as pd

from src.utils.config import CURRENT_YEAR


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with derived features added."""
    df = df.copy()
    df["car_age"] = CURRENT_YEAR - df["year"]
    df["car_age"] = df["car_age"].clip(lower=0)
    df["kms_per_year"] = df["kms_driven"] / df["car_age"].replace(0, 1)
    return df


def split_features_target(df: pd.DataFrame, target_column: str = "price"):
    """Split a DataFrame into features (X) and target (y)."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
