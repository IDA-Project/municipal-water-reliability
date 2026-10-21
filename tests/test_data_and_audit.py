from __future__ import annotations

import pandas as pd
import pytest

from water_reliability.audit import audit_dataset, reconcile_national_totals
from water_reliability.constants import TARGET_COLUMN, TARGET_SOURCE_COLUMN
from water_reliability.data import clean_raw_data, safe_percentage


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


@pytest.mark.parametrize("reliable_count,population,reported,expected_percentage,expected_class", [
    (59_990, 100_000, "59.99%", 59.99, 1),
    (60_000, 100_000, "60.00%", 60.00, 0),
    (60_010, 100_000, "60.01%", 60.01, 0),
    (599_999, 1_000_000, "59.9999%", 59.9999, 1),
])
def test_reliability_percentage_and_class_at_threshold(
    raw_frame, reliable_count, population, reported, expected_percentage, expected_class
):
    raw_frame.loc[0, "Population"] = str(population)
    raw_frame.loc[0, "Population Reliable Supply"] = str(reliable_count)
    raw_frame.loc[0, TARGET_SOURCE_COLUMN] = reported
    cleaned = clean_raw_data(raw_frame)
    calculated = safe_percentage(cleaned["Population Reliable Supply"], cleaned["Population"])

    assert calculated.loc[0] == pytest.approx(expected_percentage)
    assert cleaned.loc[0, TARGET_SOURCE_COLUMN] == pytest.approx(expected_percentage)
    assert cleaned.loc[0, TARGET_COLUMN] == expected_class


def test_audit_flags_rounding_that_changes_class_without_relabelling(raw_frame):
    raw_frame.loc[0, "Population"] = "450134"
    raw_frame.loc[0, "Population Reliable Supply"] = "268409"
    raw_frame.loc[0, TARGET_SOURCE_COLUMN] = "60%"
    cleaned = clean_raw_data(raw_frame)
    findings = audit_dataset(cleaned).invalid_values
    discrepancy = findings.loc[
        findings["check"].eq("reported and calculated percentages imply different classes at 60%")
    ]

    assert discrepancy["row_index"].tolist() == [0]
    assert discrepancy.iloc[0]["calculated_percentage"] == pytest.approx(59.628688346136926)
    assert cleaned.loc[0, TARGET_COLUMN] == 0  # Preserve the declared reported-percentage target.
