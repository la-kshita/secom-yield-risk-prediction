"""Tests for the shipped model artifact.

The artifact is the only thing the service needs at runtime, and it is committed
so a fresh clone works. These tests pin the contract the API depends on and catch
the two realistic failures: a missing file, and a file saved with a different
preprocessing layout or a different training split.
"""

import importlib.metadata as metadata
from pathlib import Path

import joblib
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = PROJECT_ROOT / "models" / "secom_logistic_regression_pipeline.joblib"

REQUIRED_KEYS = {"pipeline", "selected_name", "feature_names", "target", "provenance"}
EXPECTED_STEPS = ["impute_median", "remove_constant", "scale", "classifier"]
EXPECTED_FEATURES = 590


@pytest.fixture(scope="module")
def artifact():
    assert ARTIFACT_PATH.is_file(), (
        f"Missing artifact at {ARTIFACT_PATH}. Regenerate it with the artifact-saving "
        "cell in notebooks/train.ipynb, or restore it from version control."
    )
    return joblib.load(ARTIFACT_PATH)


def test_artifact_has_exactly_the_documented_keys(artifact):
    assert set(artifact) == REQUIRED_KEYS


def test_artifact_accepts_the_full_raw_column_set(artifact):
    feature_names = list(artifact["feature_names"])
    assert len(feature_names) == EXPECTED_FEATURES
    # Column labels must be unique, or a row could map to the wrong sensor.
    assert len(set(feature_names)) == EXPECTED_FEATURES


def test_pipeline_contains_preprocessing_and_a_classifier(artifact):
    steps = artifact["pipeline"].steps
    assert [name for name, _ in steps] == EXPECTED_STEPS
    # The service calls predict_proba; predict alone would lose the ranking.
    assert hasattr(artifact["pipeline"], "predict_proba")


def test_artifact_was_fitted_without_the_test_split(artifact):
    fit_data = artifact["provenance"]["fit_data"]
    # 1,253 rows and 83 failures: the 314-row test split was excluded from fitting.
    assert fit_data["rows"] == 1253
    assert fit_data["failures"] == 83
    assert "X_test" in fit_data["source"]


def test_artifact_records_the_selection_decision(artifact):
    provenance = artifact["provenance"]
    assert provenance["selection"]["rejected"] == "LR + correlation pruning"
    assert artifact["selected_name"] == "LR baseline"


def test_artifact_libraries_match_the_installed_environment(artifact):
    """Guard against loading a pickled pipeline under a different library version.

    requirements.txt pins these versions, so a mismatch means the environment
    drifted from the pins rather than the artifact being wrong.
    """
    recorded = artifact["provenance"]["libraries"]
    installed = {
        "scikit_learn": metadata.version("scikit-learn"),
        "pandas": metadata.version("pandas"),
        "numpy": metadata.version("numpy"),
        "joblib": metadata.version("joblib"),
    }
    assert recorded == installed
