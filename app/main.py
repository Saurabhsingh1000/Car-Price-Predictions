"""
FastAPI application for the Car Price Prediction service.

Endpoints:
    GET  /health   -> service + model liveness check
    POST /predict   -> predict a used car's selling price
    GET  /           -> serves the web UI (form)
    GET  /meta      -> known categories used to populate the UI dropdowns

Errors are handled explicitly so internal stack traces are never leaked to
API consumers.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import CarFeatures, ErrorResponse, HealthResponse, PredictionResponse
from src.models.predict import ModelNotFoundError, load_metadata, load_pipeline, predict_price
from src.utils.config import CURRENCY
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

APP_DIR = Path(__file__).resolve().parent

_model_available = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup so the first request isn't slow, and
    so a missing model artifact fails fast with a clear log message."""
    global _model_available
    try:
        load_pipeline()
        _model_available = True
        logger.info("Model loaded successfully at startup.")
    except ModelNotFoundError as exc:
        _model_available = False
        logger.warning("Startup warning: %s", exc)
    yield


app = FastAPI(
    title="Car Price Prediction API",
    description="Predicts the fair market selling price of a used car from its attributes.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return a clean 422 with readable messages instead of a raw traceback."""
    messages = [
        f"{'.'.join(str(loc) for loc in err['loc'] if loc != 'body')}: {err['msg']}"
        for err in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": "; ".join(messages)})


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request):
    metadata = load_metadata()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "companies": metadata.get("known_companies", []),
            "fuel_types": metadata.get("known_fuel_types", ["Petrol", "Diesel", "LPG"]),
            "year_min": metadata.get("year_min", 1995),
            "year_max": metadata.get("year_max", 2024),
        },
    )


@app.get("/meta", tags=["meta"])
def get_metadata() -> dict:
    """Expose known categories/ranges so a frontend can build dropdowns."""
    metadata = load_metadata()
    return {
        "companies": metadata.get("known_companies", []),
        "fuel_types": metadata.get("known_fuel_types", []),
        "year_min": metadata.get("year_min"),
        "year_max": metadata.get("year_max"),
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_model_available)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["prediction"],
)
def predict(payload: CarFeatures) -> PredictionResponse | JSONResponse:
    try:
        price = predict_price(
            name=payload.name,
            company=payload.company,
            year=payload.year,
            kms_driven=payload.kms_driven,
            fuel_type=payload.fuel_type,
        )
    except ModelNotFoundError as exc:
        logger.error("Prediction failed - model unavailable: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": "Prediction model is not available. Please try again later."},
        )
    except Exception:  # pragma: no cover - safety net, never leak internals
        logger.exception("Unexpected error during prediction")
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred while making the prediction."},
        )

    return PredictionResponse(predicted_price=price, currency=CURRENCY)
