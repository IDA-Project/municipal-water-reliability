"""Regression checks for issues found in the assessment QA."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from water_reliability.assessment import paired_test, rank_models
from water_reliability.constants import TARGET_COLUMN
from water_reliability.data import clean_raw_data
from water_reliability.pipeline import build_preprocessing_pipeline
from water_reliability.split import make_holdout_split
from water_reliability.train_models import _load_development_data
from water_reliability.models import TabularResidualClassifier, build_model_specs


def test_missing_or_impossible_target_is_not_labelled_reliable(raw_frame):
    raw_frame.loc[0, "% Population Reliable Supply"] = ""
    raw_frame.loc[1, "% Population Reliable Supply"] = "101%"
    raw_frame.loc[2, "% Population Reliable Supply"] = "-1%"
    cleaned = clean_raw_data(raw_frame)
    assert cleaned.loc[:2, TARGET_COLUMN].isna().all()
    assert not make_holdout_split(cleaned).row_index.isin([0, 1, 2]).any()


@pytest.mark.parametrize("defect", ["missing_row", "unknown_partition", "fractional_index", "duplicate_id"])
def test_partition_loader_rejects_corrupt_assignments(raw_frame, tmp_path: Path, defect):
    cleaned = clean_raw_data(raw_frame)
    assignments = make_holdout_split(cleaned)
    if defect == "missing_row":
        assignments = assignments.iloc[1:]
    elif defect == "unknown_partition":
        assignments.loc[0, "split"] = "validation"
    elif defect == "fractional_index":
        assignments["row_index"] = assignments["row_index"].astype(float)
        assignments.loc[0, "row_index"] = 0.5
    else:
        raw_frame.loc[1, "WSA Name"] = raw_frame.loc[0, "WSA Name"]
    source = tmp_path / "raw.csv"
    split = tmp_path / "split.csv"
    raw_frame.to_csv(source, index=False)
    assignments.to_csv(split, index=False)
    with pytest.raises(ValueError):
        _load_development_data(source, split)


def test_missing_population_composition_is_imputed(raw_frame):
    raw_frame.loc[0, ["Urban Population", "Rural Population"]] = ""
    cleaned = clean_raw_data(raw_frame)
    values = build_preprocessing_pipeline().fit_transform(cleaned, cleaned[TARGET_COLUMN])
    assert np.isfinite(values).all()


def test_comparator_selected_by_validation_not_test():
    summary = pd.DataFrame([
        {"model_key": "dummy", "substantive": False, "mean_validation_macro_f1": 0.99},
        {"model_key": "logistic", "substantive": True, "mean_validation_macro_f1": 0.815},
        {"model_key": "mlp", "substantive": True, "mean_validation_macro_f1": 0.796},
        {"model_key": "forest", "substantive": True, "mean_validation_macro_f1": 0.795},
        {"model_key": "deep_mlp", "substantive": True, "mean_validation_macro_f1": 0.785},
    ])
    assert rank_models(summary)[:2] == ["logistic", "mlp"]


def test_exact_mcnemar_handles_small_and_zero_discordance():
    result = paired_test(np.array([0, 1, 1]), np.array([0, 1, 0]), np.array([0, 1, 1]))
    assert result["exact_binomial_p_value"] == 1.0
    assert result["discordant_cases"] == 1
    assert result["table_rows_champion_correct_wrong_columns_comparator_correct_wrong"] == [[2, 0], [1, 0]]
    identical = paired_test(np.array([0, 1]), np.array([0, 1]), np.array([0, 1]))
    assert identical["exact_binomial_p_value"] == 1.0
    with pytest.raises(ValueError):
        paired_test(np.array([0, 1]), np.array([0]), np.array([0, 1]))


def test_neural_registry_avoids_internal_validation_after_oversampling():
    specs = {spec.key: spec.estimator for spec in build_model_specs()}
    assert specs["mlp"].early_stopping is False
    assert specs["residual_tabular_nn"].validation_fraction == 0


def test_residual_backpropagation_matches_numerical_gradients():
    # A project-written neural optimiser needs more than a fit/predict smoke test.
    model = TabularResidualClassifier(hidden_dim=3, dropout=0, alpha=0.001)
    rng = np.random.default_rng(42)
    model.parameters_ = model._initialise_parameters(2, rng)
    for name, value in model.parameters_.items():
        model.parameters_[name] = rng.uniform(0.1, 0.3, value.shape)
    X = np.array([[0.3, 0.7], [0.8, 0.2]])
    y = np.array([0, 1])
    _, cache = model._forward(X, training=False)
    analytic = model._gradients(cache, y)

    def objective():
        probability, _ = model._forward(X, training=False)
        regularization = model.alpha / 2 * sum(
            (value ** 2).sum() for name, value in model.parameters_.items() if name.startswith("W"))
        return model._binary_loss(probability, y) + regularization

    for name, parameter in model.parameters_.items():
        for index in np.ndindex(parameter.shape):
            saved = parameter[index]
            parameter[index] = saved + 1e-6
            high = objective()
            parameter[index] = saved - 1e-6
            low = objective()
            parameter[index] = saved
            assert analytic[name][index] == pytest.approx((high - low) / 2e-6, abs=1e-6)
