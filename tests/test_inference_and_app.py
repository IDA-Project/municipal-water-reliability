from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from water_reliability.constants import SAFE_RAW_FEATURES
from water_reliability.inference import load_pipeline, predict, validate_inputs

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    [100000, 80000, 60000, 50000, 40000, 30000],
    [50000, 40000, 50000, 40000, 0, 0],
    [80000, 50000, 0, 0, 80000, 50000],
]


@pytest.mark.parametrize("counts", CASES)
def test_packaged_pipeline_predicts_three_valid_cases(counts):
    model = load_pipeline(ROOT / "artifacts/water_reliability_pipeline.joblib")
    label, probability = predict(model, dict(zip(SAFE_RAW_FEATURES, counts)))
    assert label in (0, 1)
    assert 0 <= probability <= 1


@pytest.mark.parametrize("field,value", [
    ("Population", 0), ("Population", float("inf")),
    ("Population with access to water", float("nan")),
    ("Urban Population", -1), ("Rural Population", 1.5),
    ("Population", True), ("Urban Population with access to water", 70000),
    ("Population with access to water", 81000), ("Rural Population", 30000),
])
def test_invalid_counts_are_rejected(field, value):
    values = dict(zip(SAFE_RAW_FEATURES, CASES[0]))
    values[field] = value
    with pytest.raises(ValueError):
        validate_inputs(values)


@pytest.mark.parametrize("counts", CASES)
def test_streamlit_runs_complete_pipeline_for_three_valid_inputs(counts):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    assert not app.exception
    for field, value in zip(app.number_input, counts):
        field.set_value(float(value))
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 1
    assert "Predicted category:" in app.success[0].value


def test_streamlit_rejects_invalid_case_without_prediction():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    app.number_input[1].set_value(200000.0)
    app.button[0].click().run()
    assert not app.exception
    assert "cannot exceed" in app.error[0].value
    assert not app.metric


def test_corrupt_artifact_is_rejected_before_loading(tmp_path):
    original = ROOT / "artifacts/water_reliability_pipeline.joblib"
    damaged = tmp_path / original.name
    damaged.write_bytes(b"not a model")
    damaged.with_suffix(".json").write_bytes(original.with_suffix(".json").read_bytes())
    with pytest.raises(ValueError, match="checksum"):
        load_pipeline(damaged)
