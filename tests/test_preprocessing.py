"""Tests for src/data/preprocessing.py — verifies real cleaning behaviour,
not just that the module imports."""

import pandas as pd
import pytest

from src.data.preprocessing import clean_data


@pytest.fixture
def messy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [
                "Hyundai Grand i10 Magna 1.2",
                "Maruti Suzuki Alto 800 Vxi",
                "Ford EcoSport Titanium Plus",
                "Duplicate Car Listing Here",
                "Duplicate Car Listing Here",
                "Bad Year Car Listing",
                "Missing Fuel Car Listing",
                "Extreme Outlier Car Listing",
            ],
            "company": ["Hyundai", "Maruti", "Ford", "Tata", "Tata", "Tata", "Tata", "Tata"],
            "year": ["2014", "2018", "2014", "2015", "2015", "not_a_year", "2016", "2017"],
            "Price": [
                "3,25,000",
                "Ask For Price",
                "5,75,000",
                "2,00,000",
                "2,00,000",
                "1,00,000",
                "1,50,000",
                "9,000,000",
            ],
            "kms_driven": [
                "28,000 kms",
                "22,000 kms",
                "36,000 kms",
                "10,000 kms",
                "10,000 kms",
                "5,000 kms",
                "5,000 kms",
                "5,000 kms",
            ],
            "fuel_type": [
                "Petrol", "Petrol", "Diesel", "Petrol", "Petrol", "Petrol", None, "Petrol",
            ],
        }
    )


def test_clean_data_removes_ask_for_price_rows(messy_df):
    cleaned = clean_data(messy_df)
    # The "Ask For Price" row (Maruti Alto) must not survive cleaning.
    assert "Maruti" not in cleaned["company"].values
    assert not (cleaned["name"].str.contains("Alto")).any()


def test_clean_data_parses_price_and_kms_as_int(messy_df):
    cleaned = clean_data(messy_df)
    assert pd.api.types.is_integer_dtype(cleaned["price"])
    assert pd.api.types.is_integer_dtype(cleaned["kms_driven"])
    hyundai_row = cleaned[cleaned["company"] == "Hyundai"].iloc[0]
    assert hyundai_row["price"] == 325000
    assert hyundai_row["kms_driven"] == 28000


def test_clean_data_drops_non_numeric_year_rows(messy_df):
    cleaned = clean_data(messy_df)
    assert "not_a_year" not in cleaned["year"].astype(str).values
    assert "Bad Year" not in " ".join(cleaned["name"].tolist())


def test_clean_data_drops_missing_fuel_type(messy_df):
    cleaned = clean_data(messy_df)
    assert cleaned["fuel_type"].isna().sum() == 0
    assert "Missing Fuel" not in " ".join(cleaned["name"].tolist())


def test_clean_data_removes_duplicates(messy_df):
    cleaned = clean_data(messy_df)
    assert cleaned.duplicated().sum() == 0


def test_clean_data_removes_price_outliers(messy_df):
    cleaned = clean_data(messy_df)
    assert (cleaned["price"] < 6_000_000).all()


def test_clean_data_truncates_name_to_three_words(messy_df):
    cleaned = clean_data(messy_df)
    hyundai_row = cleaned[cleaned["company"] == "Hyundai"].iloc[0]
    assert hyundai_row["name"] == "Hyundai Grand i10"


def test_clean_data_resets_index(messy_df):
    cleaned = clean_data(messy_df)
    assert list(cleaned.index) == list(range(len(cleaned)))
