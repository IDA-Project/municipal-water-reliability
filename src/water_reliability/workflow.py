"""Command-line workflow producing Q2.1-Q2.4 evidence from the NIWIS CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .audit import audit_dataset, reconcile_national_totals
from .constants import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_NATIONAL_DATA_PATH,
    DEFAULT_REPORT_DIR,
    TARGET_COLUMN,
)
from .data import clean_raw_data, load_national_summary, load_raw_data, resolve_data_path
from .pipeline import build_preprocessing_pipeline, build_training_preparation_pipeline
from .split import (
    assign_province_sensitivity_groups,
    build_repeated_cv,
    make_holdout_split,
    summarise_grouped_sensitivity_cv,
    summarise_repeated_cv,
)
from .visualize import create_eda_figures, write_interpretations


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def run(
    data_path: Path,
    national_path: Path | None,
    report_dir: Path,
    artifact_dir: Path,
) -> dict[str, object]:
    resolved_data = resolve_data_path(data_path)
    raw = load_raw_data(resolved_data)
    cleaned = clean_raw_data(raw)
    audit = audit_dataset(cleaned)

    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = report_dir / "figures"

    _write_json(report_dir / "audit_summary.json", audit.summary)
    audit.data_dictionary.to_csv(report_dir / "data_dictionary.csv", index=False)
    audit.missingness.to_csv(report_dir / "missingness.csv", index=False)
    audit.invalid_values.to_csv(report_dir / "invalid_values.csv", index=False)
    audit.outliers.to_csv(report_dir / "outliers.csv", index=False)
    if national_path is not None:
        national = load_national_summary(national_path)
        reconciliation = reconcile_national_totals(cleaned, national)
        reconciliation.to_csv(report_dir / "national_reconciliation.csv", index=False)

    evidence = create_eda_figures(cleaned, figure_dir)
    _write_json(report_dir / "eda_evidence.json", evidence)
    write_interpretations(evidence, report_dir / "interpretations.md")

    assignments = make_holdout_split(cleaned)
    groups = None
    if len(cleaned) == 144:
        groups = assign_province_sensitivity_groups(cleaned)
        assignments["province_sensitivity_group"] = assignments["row_index"].map(groups)
        province_balance = (
            pd.crosstab(groups, cleaned[TARGET_COLUMN])
            .rename(columns={0: "reliable", 1: "low_reliability"})
            .reset_index()
        )
        province_balance.to_csv(report_dir / "province_class_balance.csv", index=False)
    assignments.to_csv(report_dir / "split_assignments.csv", index=False)

    development_rows = assignments.loc[assignments["split"].eq("development"), "row_index"]
    development = cleaned.loc[development_rows]
    X_development = development.drop(columns=[TARGET_COLUMN])
    y_development = development[TARGET_COLUMN].astype(int)
    summarise_repeated_cv(y_development).to_csv(
        report_dir / "repeated_cv_fold_summary.csv", index=False
    )
    if groups is not None:
        development_groups = groups.loc[development_rows]
        summarise_grouped_sensitivity_cv(development, development_groups).to_csv(
            report_dir / "province_grouped_cv_summary.csv", index=False
        )

    preprocessor = build_preprocessing_pipeline()
    transformed = preprocessor.fit_transform(X_development, y_development)
    transformed_names = preprocessor.named_steps["preprocess"].get_feature_names_out()
    pd.DataFrame({"transformed_feature": transformed_names}).to_csv(
        report_dir / "preprocessing_features.csv", index=False
    )
    joblib.dump(preprocessor, artifact_dir / "q2_preprocessor.joblib")

    training_preparation = build_training_preparation_pipeline()
    balanced_X, balanced_y = training_preparation.fit_resample(X_development, y_development)
    cv = build_repeated_cv()

    pipeline_summary = {
        "data_path": str(resolved_data),
        "development_rows": int(len(development)),
        "test_rows": int(assignments["split"].eq("test").sum()),
        "transformed_feature_count": int(transformed.shape[1]),
        "transformed_features": transformed_names.tolist(),
        "pre_resampling_class_counts": {
            str(key): int(value) for key, value in y_development.value_counts().sort_index().items()
        },
        "post_resampling_class_counts": {
            str(key): int(value) for key, value in balanced_y.value_counts().sort_index().items()
        },
        "resampled_training_rows": int(balanced_X.shape[0]),
        "cross_validation": {
            "type": type(cv).__name__,
            "splits": 5,
            "repeats": 5,
            "total_validation_folds": 25,
        },
        "leakage_controls": [
            "The target is created before splitting, using a pre-declared policy threshold.",
            "Reliability counts and percentages are not selected by the feature engineer.",
            "The WSA identifier is not selected by the feature engineer.",
            "Imputation, encoding and scaling are fitted on development/training-fold rows only.",
            "Oversampling is applied only by fit_resample on training rows.",
            "The held-out test rows are not transformed during this Q2 workflow.",
        ],
    }
    _write_json(report_dir / "q2_4_pipeline_summary.json", pipeline_summary)
    return pipeline_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--national-data", type=Path, default=DEFAULT_NATIONAL_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run(args.data, args.national_data, args.output, args.artifacts)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
