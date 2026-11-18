"""FastAPI service for the saved SECOM failure-risk model.

The service loads the fitted pipeline from models/ once, at import time, and uses it for every request. 
All preprocessing (imputation, column removal, scaling) lives inside that saved pipeline, so the API never 
transforms measurements itself.

The returned values are uncalibrated risk scores from a model trained with
class_weight="balanced". They rank rows by relative risk; they are not calibrated
failure probabilities and they are not pass/fail decisions. No decision threshold is
stored in the artifact or applied here.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Paths are resolved from this file, so the service works no matter which directory
# the command is started from. parents[1] is the repository root, because this file
# lives in src/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = PROJECT_ROOT / "models" / "secom_logistic_regression_pipeline.joblib"

if not ARTIFACT_PATH.is_file():
    raise FileNotFoundError(
        f"Model artifact not found at {ARTIFACT_PATH}. "
        "Run the artifact-saving cell in notebooks/train.ipynb first."
    )

# Loaded once at startup, not per request.
ARTIFACT = joblib.load(ARTIFACT_PATH)
PIPELINE = ARTIFACT["pipeline"]
FEATURE_NAMES = list(ARTIFACT["feature_names"])
EXPECTED_FEATURES = len(FEATURE_NAMES)

app = FastAPI(
    title="SECOM Yield Risk Prediction",
    version="1.0.0",
    description=(
        "Returns an uncalibrated risk score for each row of process measurements. "
        "Scores rank rows by relative failure risk; they are not decisions."
    ),
)


class PredictionRequest(BaseModel):
    """Request body: one or more rows of raw measurements."""

    records: list[list[float | None]] = Field(
        ...,
        description=(
            f"List of rows. Each row must contain exactly {EXPECTED_FEATURES} numeric "
            "measurements in the original SECOM column order. Use null for a missing "
            "reading; do not impute or scale anything before sending."
        ),
    )


@app.get("/")
def root() -> dict:
    """Landing page, so opening the base URL does not look like a failure."""
    return {
        "service": "SECOM Yield Risk Prediction",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "note": "Risk scores are uncalibrated rankings, not pass/fail decisions.",
    }


@app.get("/health")
def health() -> dict:
    """Report that the service is up and which model it loaded."""
    return {
        "status": "ok",
        "model": ARTIFACT["selected_name"],
        "expected_features": EXPECTED_FEATURES,
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    """Score raw measurement rows and return one risk score per row."""
    if not request.records:
        raise HTTPException(
            status_code=422,
            detail="'records' is empty. Send at least one row of measurements.",
        )

    for position, row in enumerate(request.records):
        if len(row) != EXPECTED_FEATURES:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Row {position} has {len(row)} measurements, "
                    f"but exactly {EXPECTED_FEATURES} are required."
                ),
            )

    for position, row in enumerate(request.records):
        for column, value in enumerate(row):
            if value is not None and not np.isfinite(value):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Row {position}, column {column} is not a finite number. "
                        "Send a number or null."
                    ),
                )

    # Column order comes from the artifact, so the pipeline sees the same layout it
    # was fitted on. dtype=float keeps an all-null column numeric (NaN), which the
    # pipeline's imputer handles.
    frame = pd.DataFrame(request.records, columns=FEATURE_NAMES, dtype=float)

    risk_scores = PIPELINE.predict_proba(frame)[:, 1]

    return {"risk_scores": [float(score) for score in risk_scores]}
