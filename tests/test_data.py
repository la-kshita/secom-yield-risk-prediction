"""Tests for the raw data files.

The raw data is downloaded rather than committed (it is 54 MB), so this module
skips when the files are absent. On a machine that has them, it checks the shape,
the label encoding and the missing-value representation that the notebook relies
on. If these drift, the notebook's numbers are no longer reproducible.
"""

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MEASUREMENTS = DATA_DIR / "secom.data"
LABELS = DATA_DIR / "secom_labels.data"

pytestmark = pytest.mark.skipif(
    not (MEASUREMENTS.is_file() and LABELS.is_file()),
    reason="Raw SECOM files not present; see data/README.md for the download.",
)

EXPECTED_ROWS = 1567
EXPECTED_FEATURES = 590
EXPECTED_FAILURES = 104


@pytest.fixture(scope="module")
def raw():
    features = pd.read_csv(MEASUREMENTS, sep=r"\s+", header=None)
    labels = pd.read_csv(LABELS, sep=r"\s+", header=None)
    return features, labels


def test_measurements_shape(raw):
    features, _ = raw
    assert features.shape == (EXPECTED_ROWS, EXPECTED_FEATURES)


def test_labels_are_aligned_with_measurements(raw):
    features, labels = raw
    assert len(labels) == len(features)
    assert labels.shape[1] == 2  # label + timestamp


def test_labels_use_the_documented_encoding(raw):
    _, labels = raw
    assert set(labels[0].unique()) == {-1, 1}
    assert int((labels[0] == 1).sum()) == EXPECTED_FAILURES


def test_missing_values_are_parsed_as_nan(raw):
    features, _ = raw
    assert features.isna().any().any()
    # Every column has at least one reading; the notebook's constant-column removal
    # is about variance, not about empty columns.
    assert not features.isna().all().any()
