"""Reproducible audit for class balance and common data-quality failures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import (
    COLUMN_DESCRIPTIONS,
    COUNT_COLUMNS,
    ID_COLUMN,
    PERCENT_COLUMNS,
    RELIABILITY_COLUMNS,
    TARGET_COLUMN,
    TARGET_SOURCE_COLUMN,
    TARGET_THRESHOLD,
)
from .data import safe_percentage


@dataclass(frozen=True)
class AuditResult:
    summary: dict[str, object]
    data_dictionary: pd.DataFrame
    missingness: pd.DataFrame
    invalid_values: pd.DataFrame
    outliers: pd.DataFrame


def reconcile_national_totals(frame: pd.DataFrame, national: pd.Series) -> pd.DataFrame:
    """Compare WSA sums with the separate national control totals."""
    rows: list[dict[str, object]] = []
    for column in COUNT_COLUMNS:
        wsa_sum = float(frame[column].sum(skipna=True))
        national_value = float(national[column]) if column in national.index else np.nan
        difference = wsa_sum - national_value if np.isfinite(national_value) else np.nan
        rows.append(
            {
                "measure": column,
                "wsa_sum": int(wsa_sum),
                "national_value": int(national_value) if np.isfinite(national_value) else pd.NA,
                "difference": int(difference) if np.isfinite(difference) else pd.NA,
                "matches": bool(difference == 0) if np.isfinite(difference) else False,
                "wsa_missing_count": int(frame[column].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def build_data_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column, description in COLUMN_DESCRIPTIONS.items():
        if column == ID_COLUMN:
            role = "identifier (excluded from modelling)"
        elif column == TARGET_SOURCE_COLUMN:
            role = f"target source (Class 1 when < {TARGET_THRESHOLD:g}%)"
        elif column in RELIABILITY_COLUMNS:
            role = "target-derived leakage (excluded from modelling)"
        else:
            role = "candidate raw predictor"
        rows.append(
            {
                "column": column,
                "description": description,
                "parsed_dtype": str(frame[column].dtype),
                "role": role,
            }
        )
    rows.append(
        {
            "column": TARGET_COLUMN,
            "description": (
                f"Binary label: 1 = low reliability (< {TARGET_THRESHOLD:g}%); "
                f"0 = reliable (>= {TARGET_THRESHOLD:g}%)."
            ),
            "parsed_dtype": str(frame[TARGET_COLUMN].dtype),
            "role": "model target",
        }
    )
    return pd.DataFrame(rows)


def _invalid_record(
    frame: pd.DataFrame, mask: pd.Series, check: str, columns: tuple[str, ...]
) -> pd.DataFrame:
    selected = frame.loc[mask.fillna(False), [ID_COLUMN, *columns]].copy()
    if selected.empty:
        return pd.DataFrame(columns=["row_index", ID_COLUMN, "check", *columns])
    selected.insert(0, "row_index", selected.index)
    selected.insert(2, "check", check)
    return selected


def find_invalid_values(frame: pd.DataFrame, percent_tolerance: float = 0.5) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for column in COUNT_COLUMNS:
        records.append(
            _invalid_record(frame, frame[column].lt(0), "negative count", (column,))
        )
    for column in PERCENT_COLUMNS:
        records.append(
            _invalid_record(
                frame,
                frame[column].lt(0) | frame[column].gt(100),
                "percentage outside [0, 100]",
                (column,),
            )
        )

    comparisons = (
        ("Population with access to water", "Population"),
        ("Population Reliable Supply", "Population"),
        ("Urban Population with access to water", "Urban Population"),
        ("Urban Population Reliable Supply", "Urban Population"),
        ("Rural Population with access to water", "Rural Population"),
        ("Rural Population Reliable Supply", "Rural Population"),
    )
    for part, total in comparisons:
        records.append(
            _invalid_record(frame, frame[part].gt(frame[total]), "component exceeds total", (part, total))
        )

    percentage_checks = (
        ("Population Reliable Supply", "Population", TARGET_SOURCE_COLUMN),
        (
            "Urban Population Reliable Supply",
            "Urban Population",
            "% Urban Population Reliable Supply",
        ),
        (
            "Rural Population Reliable Supply",
            "Rural Population",
            "% Rural Population Reliable Supply",
        ),
    )
    for numerator, denominator, reported in percentage_checks:
        calculated = safe_percentage(frame[numerator], frame[denominator])
        mismatch = calculated.sub(frame[reported]).abs().gt(percent_tolerance)
        subset = _invalid_record(
            frame,
            mismatch,
            f"reported percentage differs by > {percent_tolerance:g} points",
            (numerator, denominator, reported),
        )
        if not subset.empty:
            subset["calculated_percentage"] = calculated.loc[subset["row_index"]].to_numpy()
        records.append(subset)

    populated = [record for record in records if not record.empty]
    if not populated:
        return pd.DataFrame(columns=["row_index", ID_COLUMN, "check"])
    return pd.concat(populated, ignore_index=True, sort=False)


def find_iqr_outliers(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for column in (*COUNT_COLUMNS, *PERCENT_COLUMNS):
        values = frame[column].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = frame[column].lt(lower) | frame[column].gt(upper)
        if mask.any():
            subset = frame.loc[mask, [ID_COLUMN, column]].copy()
            subset.insert(0, "row_index", subset.index)
            subset.insert(2, "column", column)
            subset = subset.rename(columns={column: "value"})
            subset["lower_fence"] = lower
            subset["upper_fence"] = upper
            records.append(subset)
    if not records:
        return pd.DataFrame(
            columns=["row_index", ID_COLUMN, "column", "value", "lower_fence", "upper_fence"]
        )
    return pd.concat(records, ignore_index=True)


def audit_dataset(frame: pd.DataFrame) -> AuditResult:
    missingness = pd.DataFrame(
        {
            "column": frame.columns,
            "missing_count": [int(frame[column].isna().sum()) for column in frame.columns],
            "missing_percent": [float(frame[column].isna().mean() * 100) for column in frame.columns],
        }
    ).sort_values(["missing_percent", "column"], ascending=[False, True], ignore_index=True)

    class_counts = frame[TARGET_COLUMN].value_counts(dropna=False).sort_index()
    class_balance = {
        str(key): {"count": int(value), "percent": float(value / len(frame) * 100)}
        for key, value in class_counts.items()
    }
    invalid = find_invalid_values(frame)
    outliers = find_iqr_outliers(frame)
    duplicate_ids = frame[ID_COLUMN].duplicated(keep=False)

    summary: dict[str, object] = {
        "rows": int(len(frame)),
        "columns_after_cleaning": int(frame.shape[1]),
        "unique_wsa_names": int(frame[ID_COLUMN].nunique(dropna=True)),
        "duplicate_rows": int(frame.duplicated().sum()),
        "rows_with_duplicate_wsa_name": int(duplicate_ids.sum()),
        "target_definition": f"1 when {TARGET_SOURCE_COLUMN} < {TARGET_THRESHOLD:g}; otherwise 0",
        "class_balance": class_balance,
        "missing_cells": int(frame.isna().sum().sum()),
        "invalid_findings": int(len(invalid)),
        "iqr_outlier_findings": int(len(outliers)),
        "leakage_columns_excluded": list(RELIABILITY_COLUMNS),
        "notes": [
            "IQR flags are review prompts, not automatic deletion rules.",
            "Missing target-source rows cannot be used for stratified modelling.",
            "The WSA name is an identifier and is not encoded as a predictor.",
        ],
    }
    return AuditResult(
        summary=summary,
        data_dictionary=build_data_dictionary(frame),
        missingness=missingness,
        invalid_values=invalid,
        outliers=outliers,
    )
