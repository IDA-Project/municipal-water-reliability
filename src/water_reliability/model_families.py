"""Question 3.3: model-family selection and requirement compliance.

Template wording (Q3.3): traditional examples include Logistic Regression,
Decision Tree/Random Forest, Support Vector Machine, Naive Bayes or gradient
boosting; neural requirements are an MLP plus one task-appropriate second
architecture (for example an approved tabular neural network).

This module does not train anything. It inspects the real estimator objects in
``models.build_model_specs()`` so the family labels cannot drift from the code,
checks them against the template, and writes auditable evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .models import ModelSpec, TabularResidualClassifier, build_model_specs

DEFAULT_Q3_REPORT_DIR = Path("reports/q3")

# Estimator classes mapped to the template's own example names (Q3.3).
TEMPLATE_TRADITIONAL_EXAMPLES: dict[type, str] = {
    LogisticRegression: "Logistic Regression",
    DecisionTreeClassifier: "Decision Tree / Random Forest",
    RandomForestClassifier: "Decision Tree / Random Forest",
    SVC: "Support Vector Machine",
    GaussianNB: "Naive Bayes",
    GradientBoostingClassifier: "Gradient boosting",
    HistGradientBoostingClassifier: "Gradient boosting",
}

PENDING_APPROVAL = (
    "pending: lecturer approval of the second neural architecture is not recorded"
)


def _template_example(estimator: object) -> str | None:
    for cls, name in TEMPLATE_TRADITIONAL_EXAMPLES.items():
        if isinstance(estimator, cls):
            return name
    return None


def classify_role(spec: ModelSpec) -> tuple[str, str]:
    """Return (verified_family, template_requirement) from the estimator type."""
    estimator = spec.estimator
    if isinstance(estimator, DummyClassifier):
        return "naive baseline", "Q3.1 baseline (does not count towards the four)"
    example = _template_example(estimator)
    if example is not None:
        return "traditional", f"Template traditional example: {example}"
    if isinstance(estimator, MLPClassifier):
        return "neural", "Template requirement: MLP"
    if isinstance(estimator, TabularResidualClassifier):
        return "neural", "Template requirement: task-appropriate second architecture"
    return "unclassified", "Not a template-recognised model"


def build_compliance_table(
    specs: list[ModelSpec] | None = None, approval_evidence: str | None = None
) -> pd.DataFrame:
    specs = build_model_specs() if specs is None else specs
    rows = []
    for spec in specs:
        verified_family, requirement = classify_role(spec)
        approval = ""
        if isinstance(spec.estimator, TabularResidualClassifier):
            approval = approval_evidence or PENDING_APPROVAL
        rows.append(
            {
                "model_key": spec.key,
                "model": spec.label,
                "estimator_class": type(spec.estimator).__name__,
                "declared_family": spec.family,
                "verified_family": verified_family,
                "family_label_matches_code": spec.family == verified_family,
                "counts_as_substantive": spec.substantive,
                "template_requirement": requirement,
                "approval_status": approval,
            }
        )
    return pd.DataFrame(rows)


def check_requirements(table: pd.DataFrame) -> list[dict[str, object]]:
    """Evaluate each Q3.3 requirement; every check reports pass/fail and detail."""
    substantive = table.loc[table["counts_as_substantive"]]
    traditional = substantive.loc[substantive["verified_family"] == "traditional"]
    neural = substantive.loc[substantive["verified_family"] == "neural"]
    trad_examples = {
        _template_example(spec.estimator)
        for spec in build_model_specs()
        if spec.key in set(traditional["model_key"])
    }
    estimator_types = set(neural["estimator_class"])
    second = neural.loc[neural["estimator_class"] != "MLPClassifier"]
    baseline_counted = table.loc[
        (table["verified_family"] == "naive baseline") & table["counts_as_substantive"]
    ]
    unapproved = second.loc[
        second["approval_status"].astype(str).str.startswith("pending")
    ]

    def check(name: str, passed: bool, detail: str) -> dict[str, object]:
        return {"check": name, "passed": bool(passed), "detail": detail}

    return [
        check(
            "at_least_four_substantive_models",
            len(substantive) >= 4,
            f"{len(substantive)} substantive models",
        ),
        check(
            "at_least_two_traditional_from_template_list",
            len(traditional) >= 2 and None not in trad_examples,
            f"{len(traditional)} traditional: {sorted(map(str, trad_examples))}",
        ),
        check(
            "at_least_two_neural",
            len(neural) >= 2,
            f"{len(neural)} neural: {sorted(estimator_types)}",
        ),
        check(
            "mlp_present",
            "MLPClassifier" in estimator_types,
            "MLPClassifier required by the template",
        ),
        check(
            "second_architecture_is_distinct_from_mlp",
            len(second) >= 1,
            f"second architecture: {sorted(second['estimator_class'])}",
        ),
        check(
            "naive_baseline_not_counted",
            baseline_counted.empty,
            "baseline excluded from the substantive count",
        ),
        check(
            "declared_families_match_estimator_types",
            bool(table["family_label_matches_code"].all()),
            "family labels verified against real estimator classes",
        ),
        check(
            "second_architecture_approval_recorded",
            unapproved.empty,
            "approved" if unapproved.empty else PENDING_APPROVAL,
        ),
    ]


def write_evidence(
    report_dir: Path, approval_evidence: str | None = None
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    report_dir.mkdir(parents=True, exist_ok=True)
    table = build_compliance_table(approval_evidence=approval_evidence)
    checks = check_requirements(table)
    table.to_csv(report_dir / "q3_3_model_family_compliance.csv", index=False)
    summary = {
        "scope": "Question 3.3 model-family selection",
        "all_checks_passed": all(item["passed"] for item in checks),
        "blocking_items": [item["check"] for item in checks if not item["passed"]],
        "checks": checks,
    }
    (report_dir / "q3_3_requirement_checks.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return table, checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_Q3_REPORT_DIR)
    parser.add_argument(
        "--approval-evidence",
        default=None,
        help="Short reference (e.g. email date) proving lecturer approval of the "
        "second neural architecture. Omit while approval is outstanding.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    table, checks = write_evidence(args.output, args.approval_evidence)
    print(table[["model", "estimator_class", "verified_family", "template_requirement"]]
          .to_string(index=False))
    print()
    for item in checks:
        print(f"[{'PASS' if item['passed'] else 'OPEN'}] {item['check']}: {item['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
