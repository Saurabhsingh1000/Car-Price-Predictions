"""Pydantic request/response models for the Car Price Prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class CarFeatures(BaseModel):
    """Input payload for a price prediction request."""

    name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Car model listing name, e.g. 'Hyundai Grand i10'.",
        examples=["Hyundai Grand i10"],
    )
    company: str = Field(
        ...,
        min_length=2,
        max_length=50,
        description="Manufacturer / brand, e.g. 'Hyundai'.",
        examples=["Hyundai"],
    )
    year: int = Field(
        ...,
        ge=1990,
        le=2026,
        description="Manufacturing year of the car.",
        examples=[2016],
    )
    kms_driven: int = Field(
        ...,
        ge=0,
        le=1_000_000,
        description="Total kilometers driven.",
        examples=[30000],
    )
    fuel_type: str = Field(
        ...,
        description="Fuel type: Petrol, Diesel or LPG.",
        examples=["Petrol"],
    )

    @field_validator("fuel_type")
    @classmethod
    def validate_fuel_type(cls, value: str) -> str:
        allowed = {"petrol", "diesel", "lpg"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(
                f"fuel_type must be one of {sorted(allowed)}, got '{value}'"
            )
        return value.strip().capitalize() if normalized != "lpg" else "LPG"

    @field_validator("name", "company")
    @classmethod
    def strip_and_validate_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty or whitespace")
        return value


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Predicted selling price.")
    currency: str = Field(..., description="Currency of the predicted price.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    detail: str
