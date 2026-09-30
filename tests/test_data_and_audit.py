from __future__ import annotations

import pandas as pd

from water_reliability.audit import audit_dataset, reconcile_national_totals
from water_reliability.constants import TARGET_COLUMN
from water_reliability.data import clean_raw_data


def test_cleaning_normalises_columns_parses_numbers_and_builds_target(raw_frame: pd.DataFrame) -> None:
    cleaned = clean_raw_data(raw_frame)

    assert "Unnamed: 0" not in cleaned
    assert "Population with access to water" in cleaned
    assert cleaned.loc[1, "Population"] == 101_000
    assert pd.isna(cleaned.loc[0, "Population with access to water"])
    assert cleaned[TARGET_COLUMN].value_counts().sort_index().to_dict() == {0: 35, 1: 15}


def test_audit_reports_balance_missingness_and_no_false_percentage_errors(raw_frame: pd.DataFrame) -> None:
    cleaned = clean_raw_data(raw_frame)
    result = audit_dataset(cleaned)

    assert result.summary["rows"] == 50
    assert result.summary["duplicate_rows"] == 0
    assert result.summary["class_balance"]["1"]["count"] == 15
    missing = result.missingness.set_index("column")
    assert missing.loc["Population with access to water", "missing_count"] == 1
    assert not result.invalid_values["check"].str.contains("percentage", na=False).any()


def test_national_reconciliation_exposes_missing_wsa_total(raw_frame: pd.DataFrame) -> None:
    cleaned = clean_raw_data(raw_frame)
    national = cleaned.sum(numeric_only=True)
    national["Population with access to water"] = cleaned["Population with access to water"].fillna(90_000).sum()
    reconciliation = reconcile_national_totals(cleaned, national).set_index("measure")

    assert reconciliation.loc["Population", "matches"]
    assert not reconciliation.loc["Population with access to water", "matches"]
    assert reconciliation.loc["Population with access to water", "wsa_missing_count"] == 1
