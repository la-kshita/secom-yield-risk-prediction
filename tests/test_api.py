"""Contract tests for the prediction API.

These tests describe what a caller may rely on: the routes exist, a valid row of
measurements is scored, and malformed input is rejected instead of reaching the
model. They deliberately use synthetic rows so they do not depend on the raw data
files, which are not committed.

Importing src.app loads the model artifact at import time, so a missing or
incompatible artifact shows up here as a collection error rather than as a
mystery 500 response later.
"""

import pytest
from fastapi.testclient import TestClient

from src.app import EXPECTED_FEATURES, app

client = TestClient(app)

# A row of zeros is a valid measurement row: zeros are numbers, not missing values.
VALID_ROW = [0.0] * EXPECTED_FEATURES


def test_expected_feature_count_matches_dataset():
    # SECOM has 590 anonymous process measurements, not 591.
    assert EXPECTED_FEATURES == 590


def test_root_route_points_at_the_useful_endpoints():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["health"] == "/health"
    assert "predict" in body["predict"] or body["predict"].endswith("/predict")


def test_health_reports_the_loaded_model():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["expected_features"] == EXPECTED_FEATURES
    assert body["model"]


def test_predict_returns_one_score_per_row():
    response = client.post("/predict", json={"records": [VALID_ROW]})
    assert response.status_code == 200
    body = response.json()
    assert list(body) == ["risk_scores"]
    assert len(body["risk_scores"]) == 1
    score = body["risk_scores"][0]
    assert 0.0 <= score <= 1.0


def test_predict_handles_several_rows_in_order():
    rows = [VALID_ROW, [1.0] * EXPECTED_FEATURES]
    body = client.post("/predict", json={"records": rows}).json()
    assert len(body["risk_scores"]) == 2

    # Sending the same rows one at a time must give the same scores, in the same
    # order: no state is carried between requests.
    first = client.post("/predict", json={"records": [rows[0]]}).json()["risk_scores"][0]
    second = client.post("/predict", json={"records": [rows[1]]}).json()["risk_scores"][0]
    assert body["risk_scores"] == pytest.approx([first, second])


def test_predict_accepts_missing_readings_as_null():
    # An all-null row is extreme but must not crash: the pipeline imputes medians.
    all_missing = [None] * EXPECTED_FEATURES
    response = client.post("/predict", json={"records": [all_missing]})
    assert response.status_code == 200
    assert 0.0 <= response.json()["risk_scores"][0] <= 1.0


def test_predict_passes_nulls_through_alongside_numbers():
    partly_missing = [None, 0.0, 1.5] + [0.0] * (EXPECTED_FEATURES - 3)
    response = client.post("/predict", json={"records": [partly_missing]})
    assert response.status_code == 200


@pytest.mark.parametrize("width", [0, 1, EXPECTED_FEATURES - 1, EXPECTED_FEATURES + 1])
def test_predict_rejects_wrong_row_width(width):
    response = client.post("/predict", json={"records": [[0.0] * width]})
    assert response.status_code == 422
    assert str(EXPECTED_FEATURES) in response.json()["detail"]


def test_predict_rejects_empty_records():
    response = client.post("/predict", json={"records": []})
    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


def test_predict_rejects_missing_records_key():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_rejects_malformed_json():
    response = client.post(
        "/predict",
        content='{"records": [[0.0, 1.0',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
