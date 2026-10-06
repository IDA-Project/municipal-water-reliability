from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

from water_reliability.model_families import (
    PENDING_APPROVAL,
    build_compliance_table,
    check_requirements,
    write_evidence,
)
from water_reliability.models import ModelSpec, build_model_specs


def _results(table, **kw):
    return {c["check"]: c["passed"] for c in check_requirements(table, **kw)}


def test_current_model_set_meets_q3_3_except_pending_approval() -> None:
    results = _results(build_compliance_table())
    approval = results.pop("second_architecture_approval_recorded")

    assert all(results.values())
    assert approval is False


def test_approval_evidence_clears_the_open_item() -> None:
    table = build_compliance_table(approval_evidence="Email from lecturer, 2026-10-01")

    assert all(_results(table).values())


def test_missing_second_neural_model_is_detected() -> None:
    specs = [s for s in build_model_specs() if s.key != "residual_tabular_nn"]
    results = _results(build_compliance_table(specs))

    assert results["at_least_four_substantive_models"] is False
    assert results["at_least_two_neural"] is False
    assert results["second_architecture_is_distinct_from_mlp"] is False


def test_mislabelled_family_and_counted_baseline_are_detected() -> None:
    specs = build_model_specs()
    specs.append(
        ModelSpec("svm", "SVM", "neural", True, SVC(), "wrongly labelled")
    )
    specs.append(
        ModelSpec("dummy2", "Dummy", "naive baseline", True, DummyClassifier(), "x")
    )
    results = _results(build_compliance_table(specs))

    assert results["declared_families_match_estimator_types"] is False
    assert results["naive_baseline_not_counted"] is False


def test_pending_status_is_written_and_evidence_files_exist(tmp_path) -> None:
    table, checks = write_evidence(tmp_path)

    assert (tmp_path / "q3_3_model_family_compliance.csv").is_file()
    assert (tmp_path / "q3_3_requirement_checks.json").is_file()
    assert PENDING_APPROVAL in table["approval_status"].tolist()
