from pathlib import Path
from xml.etree import ElementTree

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


def assert_prediction_display(app, predicted, unreliable_probability):
    card = ElementTree.fromstring(app.get("html")[0].proto.body)
    badge = card.find("./span[@class='prediction-badge']")
    primary = card.find("./div[@class='reliable-probability']")
    secondary = card.find("./div[@class='unreliable-probability']/strong")
    progress = card.find("./div[@role='progressbar']")
    category = "UNRELIABLE" if predicted == 1 else "RELIABLE"
    color = "#ef4444" if predicted == 1 else "#16a34a"
    reliable_probability = 1 - unreliable_probability

    assert badge.text == f"Predicted: {category}"
    assert f"color: {color};" in badge.attrib["style"]
    assert f"color: {color};" in primary.attrib["style"]
    assert primary.text == f"{reliable_probability:.1%}"
    assert secondary.text == f"{unreliable_probability:.1%}"
    assert float(primary.text.rstrip("%")) + float(secondary.text.rstrip("%")) == pytest.approx(100)
    assert progress.attrib["aria-label"] == "Probability of Reliable"
    assert float(progress.attrib["aria-valuenow"]) == pytest.approx(reliable_probability * 100)
    assert f"background-color: {color};" in progress[0].attrib["style"]
    width = progress[0].attrib["style"].split("width: ")[1].split("%")[0]
    assert float(width) == pytest.approx(reliable_probability * 100)


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
    model = load_pipeline(ROOT / "artifacts/water_reliability_pipeline.joblib")
    label, probability = predict(model, dict(zip(SAFE_RAW_FEATURES, counts)))
    assert_prediction_display(app, label, probability)


@pytest.mark.parametrize("counts,expected_class,expected_probability", [
    (CASES[0], 1, 0.8323830174877495),
    ([100000, 100000, 60000, 60000, 40000, 40000], 0, 0.13553213561964234),
])
def test_screenshot_predictions_and_display(counts, expected_class, expected_probability):
    model = load_pipeline(ROOT / "artifacts/water_reliability_pipeline.joblib")
    label, probability = predict(model, dict(zip(SAFE_RAW_FEATURES, counts)))
    assert label == expected_class
    assert probability == pytest.approx(expected_probability)

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    for field, value in zip(app.number_input, counts):
        field.set_value(float(value))
    app.button[0].click().run()

    assert not app.exception
    assert not app.error
    assert_prediction_display(app, expected_class, expected_probability)
    assert any("not the reliable-supply percentage" in caption.value for caption in app.caption)


@pytest.mark.parametrize("predicted,unreliable_probability", [
    (0, 0.136), (1, 0.6),  # Requested examples: 86.4% green and 40.0% red.
    (0, 0.45), (1, 0.55),  # The 60% supply rule must not determine the display category.
    (0, 0.0), (1, 1.0),  # Empty and full progress bars.
])
def test_probability_display_uses_predicted_class(monkeypatch, predicted, unreliable_probability):
    monkeypatch.setattr(
        "water_reliability.inference.predict",
        lambda model, values: (predicted, unreliable_probability),
    )
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert_prediction_display(app, predicted, unreliable_probability)


def test_streamlit_rejects_invalid_case_without_prediction():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    app.number_input[1].set_value(200000.0)
    app.button[0].click().run()
    assert not app.exception
    assert "cannot exceed" in app.error[0].value
    assert not app.get("html")


def test_corrupt_artifact_is_rejected_before_loading(tmp_path):
    original = ROOT / "artifacts/water_reliability_pipeline.joblib"
    damaged = tmp_path / original.name
    damaged.write_bytes(b"not a model")
    damaged.with_suffix(".json").write_bytes(original.with_suffix(".json").read_bytes())
    with pytest.raises(ValueError, match="checksum"):
        load_pipeline(damaged)
