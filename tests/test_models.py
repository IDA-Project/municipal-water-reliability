from __future__ import annotations

import numpy as np
import pandas as pd

from water_reliability.constants import TARGET_COLUMN
from water_reliability.data import clean_raw_data
from water_reliability.models import build_model_pipeline, build_model_specs


def test_model_registry_meets_q3_1_and_q3_2_counts() -> None:
    specs = build_model_specs()

    assert len(specs) == 5
    assert sum(not spec.substantive for spec in specs) == 1
    assert sum(spec.substantive for spec in specs) == 4
    assert sum(spec.family == "traditional" for spec in specs) == 2
    assert sum(spec.family == "neural" for spec in specs) == 2
    assert specs[0].key == "majority_baseline"


def test_every_model_pipeline_fits_and_predicts(raw_frame: pd.DataFrame) -> None:
    cleaned = clean_raw_data(raw_frame)
    X = cleaned.drop(columns=[TARGET_COLUMN])
    y = cleaned[TARGET_COLUMN].astype(int)

    for spec in build_model_specs():
        pipeline = build_model_pipeline(spec)
        pipeline.fit(X, y)
        predictions = pipeline.predict(X.iloc[:5])

        assert predictions.shape == (5,)
        assert set(np.unique(predictions)).issubset({0, 1})
        if spec.substantive:
            assert "balance" in pipeline.named_steps
        else:
            assert "balance" not in pipeline.named_steps


def test_residual_model_probabilities_are_finite(raw_frame: pd.DataFrame) -> None:
    cleaned = clean_raw_data(raw_frame)
    X = cleaned.drop(columns=[TARGET_COLUMN])
    y = cleaned[TARGET_COLUMN].astype(int)
    residual_spec = next(
        spec for spec in build_model_specs() if spec.key == "residual_tabular_nn"
    )
    pipeline = build_model_pipeline(residual_spec).fit(X, y)

    probabilities = pipeline.predict_proba(X.iloc[:7])

    assert probabilities.shape == (7, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
