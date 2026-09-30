"""Train the Q3.1 baseline and Q3.2 model set on development data only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from .constants import DEFAULT_DATA_PATH, ID_COLUMN, RANDOM_STATE, TARGET_COLUMN
from .data import clean_raw_data, load_raw_data, resolve_data_path
from .models import ModelSpec, build_model_pipeline, build_model_specs
from .split import build_repeated_cv

DEFAULT_ASSIGNMENTS_PATH = Path("reports/q2/split_assignments.csv")
DEFAULT_Q3_REPORT_DIR = Path("reports/q3")
DEFAULT_Q3_ARTIFACT_DIR = Path("models/q3")


def _serialise_parameter(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [_serialise_parameter(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _model_record(spec: ModelSpec, artifact_path: Path) -> dict[str, object]:
    parameters = {
        name: _serialise_parameter(value)
        for name, value in spec.estimator.get_params(deep=False).items()
    }
    return {
        "key": spec.key,
        "label": spec.label,
        "family": spec.family,
        "substantive": spec.substantive,
        "description": spec.description,
        "estimator": type(spec.estimator).__name__,
        "parameters": parameters,
        "artifact": str(artifact_path),
    }


def _load_development_data(
    data_path: Path, assignments_path: Path
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, Path]:
    resolved_data = resolve_data_path(data_path)
    cleaned = clean_raw_data(load_raw_data(resolved_data))
    if not assignments_path.is_file():
        raise FileNotFoundError(
            f"Split assignments not found at '{assignments_path}'. Run the Q2 workflow first."
        )
    assignments = pd.read_csv(assignments_path)
    required = {"row_index", ID_COLUMN, TARGET_COLUMN, "split"}
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError(f"Split assignments are missing columns: {missing}")

    row_index = assignments["row_index"].astype(int)
    if row_index.duplicated().any():
        raise ValueError("Split assignments contain duplicate row indices.")
    if not set(row_index).issubset(set(cleaned.index)):
        raise ValueError("Split assignments refer to rows outside the cleaned dataset.")

    reference = cleaned.loc[row_index, [ID_COLUMN, TARGET_COLUMN]].reset_index(drop=True)
    observed = assignments[[ID_COLUMN, TARGET_COLUMN]].reset_index(drop=True)
    ids_match = reference[ID_COLUMN].astype(str).equals(
        observed[ID_COLUMN].astype(str)
    )
    targets_match = reference[TARGET_COLUMN].astype(int).equals(
        observed[TARGET_COLUMN].astype(int)
    )
    if not ids_match or not targets_match:
        raise ValueError("Split assignments do not match the current dataset IDs and targets.")

    development_assignment = assignments.loc[
        assignments["split"].eq("development")
    ].copy()
    test_assignment = assignments.loc[assignments["split"].eq("test")].copy()
    if development_assignment.empty or test_assignment.empty:
        raise ValueError("Both development and test assignments are required.")
    if set(development_assignment["row_index"]).intersection(test_assignment["row_index"]):
        raise ValueError("Development and test assignments overlap.")

    development_index = development_assignment["row_index"].astype(int)
    development = cleaned.loc[development_index]
    X_development = development.drop(columns=[TARGET_COLUMN])
    y_development = development[TARGET_COLUMN].astype(int)
    return X_development, y_development, assignments, resolved_data


def train(
    data_path: Path,
    assignments_path: Path,
    report_dir: Path,
    artifact_dir: Path,
) -> pd.DataFrame:
    X_development, y_development, assignments, resolved_data = _load_development_data(
        data_path, assignments_path
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    split_indices = list(
        build_repeated_cv().split(np.zeros(len(y_development)), y_development)
    )
    summaries: list[dict[str, object]] = []
    model_records: list[dict[str, object]] = []

    for spec in build_model_specs():
        pipeline = build_model_pipeline(spec)
        scores = cross_validate(
            pipeline,
            X_development,
            y_development,
            cv=split_indices,
            scoring="f1_macro",
            return_train_score=True,
            n_jobs=1,
            error_score="raise",
        )

        pipeline.fit(X_development, y_development)
        artifact_path = artifact_dir / f"{spec.key}.joblib"
        joblib.dump(pipeline, artifact_path)
        model_records.append(_model_record(spec, artifact_path))
        summaries.append(
            {
                "model_key": spec.key,
                "model": spec.label,
                "family": spec.family,
                "substantive": spec.substantive,
                "cv_fits": len(scores["test_score"]),
                "mean_train_macro_f1": float(np.mean(scores["train_score"])),
                "mean_validation_macro_f1": float(np.mean(scores["test_score"])),
                "std_validation_macro_f1": float(np.std(scores["test_score"], ddof=1)),
                "artifact": str(artifact_path),
            }
        )

    summary = pd.DataFrame(summaries)
    summary.to_csv(report_dir / "development_cv_training_summary.csv", index=False)

    ordered_ids = "\n".join(X_development[ID_COLUMN].astype(str).sort_values())
    manifest = {
        "scope": "Questions 3.1 and 3.2 only",
        "random_state": RANDOM_STATE,
        "data_path": str(resolved_data),
        "split_assignments": str(assignments_path),
        "development_rows": int(len(X_development)),
        "development_class_counts": {
            str(key): int(value)
            for key, value in y_development.value_counts().sort_index().items()
        },
        "development_ids_sha256": hashlib.sha256(ordered_ids.encode("utf-8")).hexdigest(),
        "test_rows_reserved": int(assignments["split"].eq("test").sum()),
        "test_rows_used_for_training": False,
        "test_predictions_created": False,
        "test_metrics_computed": False,
        "validation_protocol": {
            "type": "RepeatedStratifiedKFold",
            "splits": 5,
            "repeats": 5,
            "identical_split_indices_for_every_model": True,
            "primary_sanity_score": "macro-F1",
        },
        "models": model_records,
        "approval_note": (
            "The residual tabular neural network is the task-appropriate second neural "
            "architecture selected for implementation; lecturer approval remains to be recorded."
        ),
    }
    (report_dir / "model_training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_Q3_REPORT_DIR)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_Q3_ARTIFACT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = train(args.data, args.assignments, args.output, args.artifacts)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
