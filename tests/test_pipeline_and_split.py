from __future__ import annotations

import numpy as np
import pandas as pd

from water_reliability.constants import RELIABILITY_COLUMNS, TARGET_COLUMN
from water_reliability.data import clean_raw_data
from water_reliability.pipeline import (
    build_preprocessing_pipeline,
    build_training_preparation_pipeline,
)
from water_reliability.split import build_repeated_cv, make_holdout_split


def test_holdout_is_deterministic_stratified_and_disjoint(raw_frame: pd.DataFrame) -> None:
    cleaned = clean_raw_data(raw_frame)
    first = make_holdout_split(cleaned)
    second = make_holdout_split(cleaned)

    pd.testing.assert_frame_equal(first, second)
    assert first["split"].value_counts().to_dict() == {"development": 40, "test": 10}
    assert first.loc[first["split"].eq("test"), TARGET_COLUMN].sum() == 3
    assert set(first.loc[first["split"].eq("development"), "row_index"]).isdisjoint(
        set(first.loc[first["split"].eq("test"), "row_index"])
    )


def test_preprocessor_is_invariant_to_excluded_reliability_columns(raw_frame: pd.DataFrame) -> None:
    cleaned = clean_raw_data(raw_frame)
    X = cleaned.drop(columns=[TARGET_COLUMN])
    y = cleaned[TARGET_COLUMN].astype(int)
    changed = X.copy()
    for column in RELIABILITY_COLUMNS:
        changed[column] = 999_999

    pipeline = build_preprocessing_pipeline()
    original_values = pipeline.fit_transform(X, y)
    changed_values = pipeline.transform(changed)
    np.testing.assert_allclose(original_values, changed_values)
    assert np.isfinite(original_values).all()


def test_training_only_resampling_balances_classes(raw_frame: pd.DataFrame) -> None:
    cleaned = clean_raw_data(raw_frame)
    X = cleaned.drop(columns=[TARGET_COLUMN])
    y = cleaned[TARGET_COLUMN].astype(int)
    pipeline = build_training_preparation_pipeline()
    X_balanced, y_balanced = pipeline.fit_resample(X, y)

    assert y_balanced.value_counts().to_dict() == {1: 35, 0: 35}
    assert X_balanced.shape[0] == 70
    assert np.isfinite(X_balanced).all()


def test_repeated_cv_contract() -> None:
    cv = build_repeated_cv()
    assert cv.get_n_splits() == 25
