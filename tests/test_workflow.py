from __future__ import annotations

from pathlib import Path

import pandas as pd

from water_reliability.workflow import run


def test_end_to_end_workflow_writes_required_evidence(
    raw_frame: pd.DataFrame, tmp_path: Path
) -> None:
    raw_path = tmp_path / "NIWIS.csv"
    national_path = tmp_path / "national.csv"
    report_dir = tmp_path / "reports"
    artifact_dir = tmp_path / "models"
    raw_frame.to_csv(raw_path, index=False)

    cleaned_columns = {
        "'Category": [
            "'Population",
            "'Population with access to water",
            "'Population Reliability Supply",
            "'Urban Population",
            "'Urban Population with access to water",
            "'Urban Population Reliability Supply",
            "'Rural Population",
            "'Rural Population with access to water",
            "'Rural Population Reliability Supply",
        ],
        "'Population:": [
            6_225_000,
            5_500_000,
            4_300_000,
            3_735_000,
            3_540_000,
            2_580_000,
            2_490_000,
            1_960_000,
            1_720_000,
        ],
    }
    pd.DataFrame(cleaned_columns).to_csv(national_path, index=False)

    summary = run(raw_path, national_path, report_dir, artifact_dir)

    assert summary["development_rows"] == 40
    assert summary["test_rows"] == 10
    assert (report_dir / "audit_summary.json").is_file()
    assert (report_dir / "national_reconciliation.csv").is_file()
    assert (report_dir / "split_assignments.csv").is_file()
    assert (report_dir / "repeated_cv_fold_summary.csv").is_file()
    assert (report_dir / "preprocessing_features.csv").is_file()
    assert len(list((report_dir / "figures").glob("figure_*.png"))) == 5
    assert (artifact_dir / "q2_preprocessor.joblib").is_file()
