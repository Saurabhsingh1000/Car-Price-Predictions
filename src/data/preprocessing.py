"""
Data loading and cleaning for the used-car dataset.

The raw dataset (``data/raw/car_dataset.csv``) is scraped listing data and is
messy in several well-defined ways:

* ``year`` sometimes contains non-numeric placeholder values.
* ``Price`` is a string, sometimes literally ``"Ask For Price"``, and uses
  comma thousand-separators (Indian numbering, e.g. ``"4,25,000"``).
* ``kms_driven`` is a string such as ``"45,000 kms"``.
* ``fuel_type`` has missing values.
* ``name`` is a long, inconsistent free-text listing title.

This module turns that raw data into a clean, typed DataFrame ready for
feature engineering and modelling. All cleaning steps are deterministic and
side-effect free (no filesystem I/O beyond the explicit load/save helpers),
so the exact same logic can be reused at inference time if ever needed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import (
    MAX_REASONABLE_PRICE,
    MIN_REASONABLE_PRICE,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

REQUIRED_RAW_COLUMNS = ["name", "company", "year", "Price", "kms_driven", "fuel_type"]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw scraped CSV file."""
    logger.info("Loading raw data from %s", path)
    df = pd.read_csv(path)

    missing_cols = set(REQUIRED_RAW_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Raw dataset is missing expected columns: {missing_cols}")

    logger.info("Loaded raw data with shape %s", df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning pipeline and return a clean, typed DataFrame.

    Steps (in order):
      1. Drop exact duplicate rows.
      2. Keep only rows where ``year`` is numeric, cast to int.
      3. Drop rows where ``Price`` is the placeholder ``"Ask For Price"``.
      4. Parse ``Price`` (strip commas) into an integer.
      5. Parse ``kms_driven`` (strip " kms" suffix and commas) into an integer.
      6. Drop rows with missing ``fuel_type``.
      7. Truncate ``name`` to its first 3 words for consistent categories.
      8. Remove unreasonable price outliers.
      9. Reset the index.
    """
    df = df.copy()
    n_start = len(df)

    # 1. Duplicates
    df = df.drop_duplicates()

    # 2. Year must be numeric
    df = df[df["year"].astype(str).str.isnumeric()]
    df["year"] = df["year"].astype(int)

    # 3-4. Price
    df = df[df["Price"] != "Ask For Price"]
    df["Price"] = (
        df["Price"].astype(str).str.replace(",", "", regex=False).astype(int)
    )

    # 5. kms_driven: "45,000 kms" -> 45000
    df["kms_driven"] = (
        df["kms_driven"]
        .astype(str)
        .str.split(" ")
        .str.get(0)
        .str.replace(",", "", regex=False)
    )
    df = df[df["kms_driven"].str.isnumeric()]
    df["kms_driven"] = df["kms_driven"].astype(int)

    # 6. fuel_type must be present
    df = df[~df["fuel_type"].isna()]

    # 7. Keep first 3 words of the listing name -> approximate car model
    df["name"] = df["name"].astype(str).str.split(" ").str.slice(0, 3).str.join(" ")

    # 8. Remove unreasonable price outliers (kept consistent with original EDA)
    df = df[(df["Price"] < MAX_REASONABLE_PRICE) & (df["Price"] >= MIN_REASONABLE_PRICE)]

    # 9. Also guard against nonsensical kms values (e.g. 0 is fine, but huge
    # values beyond realistic odometer readings are dropped as data errors).
    df = df[df["kms_driven"] < 1_000_000]

    df = df.reset_index(drop=True)
    df = df.rename(columns={"Price": "price"})

    n_end = len(df)
    logger.info(
        "Cleaned data: %d -> %d rows (%d removed as invalid/duplicate/outlier)",
        n_start,
        n_end,
        n_start - n_end,
    )
    return df


def save_processed_data(df: pd.DataFrame, path: Path = PROCESSED_DATA_PATH) -> None:
    """Persist the cleaned dataset to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved processed data to %s (%d rows)", path, len(df))


def load_and_clean(
    raw_path: Path = RAW_DATA_PATH, save: bool = True
) -> pd.DataFrame:
    """Convenience wrapper: load raw data, clean it, optionally persist it."""
    raw_df = load_raw_data(raw_path)
    clean_df = clean_data(raw_df)
    if save:
        save_processed_data(clean_df)
    return clean_df


if __name__ == "__main__":
    load_and_clean()
