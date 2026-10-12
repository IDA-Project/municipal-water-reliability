"""Shared Q4–Q6 evidence runner. Development decisions precede test scoring.

The original team's test results are already disclosed. A QA rerun on that
partition is a reproduction check, not a newly untouched external evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay, log_loss,
)
from sklearn.model_selection import GridSearchCV, cross_validate, learning_curve, train_test_split

from .constants import DEFAULT_DATA_PATH, ID_COLUMN, RANDOM_STATE, TARGET_COLUMN
from .data import clean_raw_data, load_raw_data
from .models import build_model_pipeline, build_model_specs
from .pipeline import build_preprocessing_pipeline
from .split import build_repeated_cv
from .train_models import DEFAULT_ASSIGNMENTS_PATH, _load_development_data

SCORING = {"accuracy": "accuracy", "macro_f1": "f1_macro", "weighted_f1": "f1_weighted",
           "recall": "recall", "precision": "precision", "roc_auc": "roc_auc",
           "average_precision": "average_precision"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def rank_models(summary: pd.DataFrame) -> list[str]:
    """Select champion and comparator only from development validation scores."""
    substantive = summary.loc[summary["substantive"].eq(True)]
    return substantive.sort_values(
        ["mean_validation_macro_f1", "model_key"], ascending=[False, True]
    )["model_key"].tolist()


def paired_test(y: np.ndarray, champion: np.ndarray, comparator: np.ndarray) -> dict:
    y, champion, comparator = map(np.asarray, (y, champion, comparator))
    if y.ndim != 1 or champion.shape != y.shape or comparator.shape != y.shape or not len(y):
        raise ValueError("Paired predictions must match the same nonempty ordered cases.")
    a, b = champion == y, comparator == y
    both = int((a & b).sum())
    champion_only = int((a & ~b).sum())
    comparator_only = int((~a & b).sum())
    neither = int((~a & ~b).sum())
    discordant = champion_only + comparator_only
    p = float(binomtest(champion_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {"table_rows_champion_correct_wrong_columns_comparator_correct_wrong":
            [[both, champion_only], [comparator_only, neither]],
            "discordant_cases": discordant, "exact_binomial_p_value": p,
            "alpha": 0.05, "reject_equal_error_rates": p < 0.05,
            "null": "Equal marginal error probabilities on paired cases",
            "alternative": "Different marginal error probabilities on paired cases",
            "interpretation": "Failure to reject does not establish equivalence or equal performance."}


def metric_values(y, prediction, probability) -> dict:
    return {"accuracy": accuracy_score(y, prediction),
            "macro_f1": f1_score(y, prediction, average="macro", labels=[0, 1], zero_division=0),
            "weighted_f1": f1_score(y, prediction, average="weighted", zero_division=0),
            "precision_class_1": precision_score(y, prediction, zero_division=0),
            "recall_class_1": recall_score(y, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y, probability),
            "average_precision": average_precision_score(y, probability)}


def uncertainty(y, prediction, samples: int = 500) -> dict:
    """Seeded case bootstrap; conditional on these WSAs and their class counts."""
    y, prediction = np.asarray(y), np.asarray(prediction)
    rng = np.random.default_rng(RANDOM_STATE)
    by_class = [np.flatnonzero(y == c) for c in [0, 1]]
    scores = []
    for _ in range(samples):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in by_class])
        scores.append(f1_score(y[indices], prediction[indices], average="macro", labels=[0, 1], zero_division=0))
    positives = y == 1
    recall_ci = binomtest(int((prediction[positives] == 1).sum()), int(positives.sum())).proportion_ci()
    return {"macro_f1_95_percentile_interval": np.quantile(scores, [0.025, 0.975]).tolist(),
            "class_1_recall_95_exact_interval": [recall_ci.low, recall_ci.high],
            "bootstrap_samples": samples, "seed": RANDOM_STATE,
            "limitation": "Small-sample, conditional intervals; spatial dependence is not accounted for."}


def search_space(key: str) -> dict:
    # The search is fixed before any QA test evaluation and tunes one family only.
    return {
        "logistic_regression": {"model__C": [0.01, 0.1, 1, 10],
                                "model__l1_ratio": [0, 1]},
        "random_forest": {"model__min_samples_leaf": [1, 2, 4],
                          "model__max_depth": [None, 5]},
        "mlp": {"model__alpha": [0.001, 0.01],
                "model__hidden_layer_sizes": [(32, 16), (16,)]},
        "residual_tabular_nn": {"model__alpha": [0.001, 0.01],
                                "model__dropout": [0.0, 0.1]},
    }[key]


def neural_diagnostics(spec, X, y, output: Path) -> None:
    """Split original development WSAs BEFORE preprocessing and oversampling.

    Curves are diagnostics on a fixed development split, not new test results
    or scores for the fitted full-development artifact.
    """
    train_X, val_X, train_y, val_y = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    preparation = build_preprocessing_pipeline()
    train_values = preparation.fit_transform(train_X, train_y)
    val_values = preparation.transform(val_X)
    train_values, balanced_y = RandomOverSampler(random_state=RANDOM_STATE).fit_resample(train_values, train_y)
    model = clone(spec.estimator)
    if spec.key == "mlp":
        rows = []
        for epoch in range(model.max_iter):
            model.partial_fit(train_values, balanced_y, classes=np.array([0, 1]))
            train_prob = model.predict_proba(train_values)
            val_prob = model.predict_proba(val_values)
            rows.append({"epoch": epoch + 1,
                         "train_loss": log_loss(balanced_y, train_prob, labels=[0, 1]),
                         "validation_loss": log_loss(val_y, val_prob, labels=[0, 1]),
                         "train_macro_f1": f1_score(balanced_y, model.predict(train_values), average="macro"),
                         "validation_macro_f1": f1_score(val_y, model.predict(val_values), average="macro")})
        history = pd.DataFrame(rows)
    else:
        model.fit(train_values, balanced_y, validation_data=(val_values, val_y.to_numpy()))
        history = pd.DataFrame({"epoch": range(1, model.n_iter_ + 1),
                                "train_loss": model.loss_curve_, "validation_loss": model.validation_loss_curve_,
                                "train_macro_f1": model.training_macro_f1_curve_,
                                "validation_macro_f1": model.validation_macro_f1_curve_})
    history.to_csv(output / f"{spec.key}_neural_history.csv", index=False)
    write_json(output / f"{spec.key}_neural_diagnostic_split.json",
               {"training_ids": train_X[ID_COLUMN].tolist(), "validation_ids": val_X[ID_COLUMN].tolist(),
                "preprocessing_fit_on_training_only": True, "oversampling_after_split": True,
                "scope": "Separate development diagnostic; no held-out test rows"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, ylabel in [(axes[0], "loss", "Binary log loss"), (axes[1], "macro_f1", "Macro-F1")]:
        for split in ["train", "validation"]:
            ax.plot(history.epoch, history[f"{split}_{metric}"], label=split)
        ax.set(xlabel="Epoch", ylabel=ylabel); ax.legend()
    fig.suptitle(spec.label + " development diagnostics"); fig.tight_layout()
    fig.savefig(output / f"{spec.key}_neural_history.png"); plt.close(fig)


def develop(data: Path, assignments: Path, output: Path, artifacts: Path, jobs: int = 2) -> dict:
    if (output / "locked_manifest.json").exists():
        raise FileExistsError("Development decisions are already locked. Reuse the recorded evidence.")
    X, y, _, resolved = _load_development_data(data, assignments)
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    folds = list(build_repeated_cv().split(X, y))
    specs = build_model_specs()
    records, score_rows = [], []
    models = {}
    for spec in specs:
        pipeline = build_model_pipeline(spec)
        scores = cross_validate(pipeline, X, y, cv=folds, scoring=SCORING,
                                return_train_score=True, n_jobs=jobs, error_score="raise")
        row = {"model_key": spec.key, "model": spec.label, "substantive": spec.substantive}
        for metric in SCORING:
            row[f"mean_validation_{metric}"] = float(scores[f"test_{metric}"].mean())
            row[f"sd_validation_{metric}"] = float(scores[f"test_{metric}"].std(ddof=1))
            row[f"mean_train_{metric}"] = float(scores[f"train_{metric}"].mean())
        records.append(row)
        for i in range(len(folds)):
            score_rows.append({"model_key": spec.key, "fold": i + 1,
                               **{k: float(v[i]) for k, v in scores.items() if k.startswith(("test_", "train_"))}})
        pipeline.fit(X, y)
        models[spec.key] = pipeline
        joblib.dump(pipeline, artifacts / f"{spec.key}.joblib")
        if spec.substantive:
            if spec.family == "neural":
                neural_diagnostics(spec, X, y, output)
            sizes, train_scores, val_scores = learning_curve(
                build_model_pipeline(spec), X, y, cv=folds[:5],
                train_sizes=[0.4, 0.7, 1.0], scoring="f1_macro", shuffle=True,
                random_state=RANDOM_STATE, n_jobs=jobs, error_score="raise")
            curve = pd.DataFrame({"training_rows": sizes, "train_macro_f1": train_scores.mean(axis=1),
                                  "validation_macro_f1": val_scores.mean(axis=1),
                                  "validation_sd": val_scores.std(axis=1, ddof=1)})
            curve.to_csv(output / f"{spec.key}_learning_curve.csv", index=False)
            fig, ax = plt.subplots()
            ax.plot(sizes, curve.train_macro_f1, label="Training")
            ax.errorbar(sizes, curve.validation_macro_f1, yerr=curve.validation_sd, label="Validation (SD)")
            ax.set(xlabel="Training WSAs", ylabel="Macro-F1", title=spec.label + " learning curve", ylim=(0, 1.05))
            ax.legend(); fig.tight_layout(); fig.savefig(output / f"{spec.key}_learning_curve.png"); plt.close(fig)
    summary = pd.DataFrame(records)
    summary.to_csv(output / "development_comparison.csv", index=False)
    pd.DataFrame(score_rows).to_csv(output / "development_fold_scores.csv", index=False)
    champion, comparator = rank_models(summary)[:2]
    original = models[champion]
    search = GridSearchCV(clone(original), search_space(champion), scoring="f1_macro",
                          cv=folds, n_jobs=jobs, return_train_score=True, error_score="raise")
    search.fit(X, y)
    pd.DataFrame(search.cv_results_).to_csv(output / "optimization_search.csv", index=False)
    optimized = search.best_estimator_
    joblib.dump(optimized, artifacts / "optimized_champion.joblib")
    ablations = []
    for name in ["full", "without_derived_features", "without_scaling", "without_oversampling"]:
        pipeline = clone(optimized)
        if name == "without_derived_features":
            processor = pipeline.named_steps["preprocess"]
            raw_features = ["population", "population_with_access", "urban_population",
                            "urban_population_with_access", "rural_population", "rural_population_with_access"]
            processor.transformers = [("numeric", processor.transformers[0][1], raw_features)]
        elif name == "without_scaling":
            pipeline.set_params(preprocess__numeric__scale="passthrough")
        elif name == "without_oversampling":
            pipeline.set_params(balance="passthrough")
        scores = cross_validate(pipeline, X, y, cv=folds, scoring="f1_macro", n_jobs=jobs, error_score="raise")
        ablations.append({"variant": name, "mean_macro_f1": float(scores["test_score"].mean()),
                          "sd_macro_f1": float(scores["test_score"].std(ddof=1))})
    ablation = pd.DataFrame(ablations)
    ablation["delta_vs_full"] = ablation.mean_macro_f1 - ablation.mean_macro_f1.iloc[0]
    ablation.to_csv(output / "ablations.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(summary.model, summary.mean_validation_macro_f1, xerr=summary.sd_validation_macro_f1)
    ax.set(xlabel="Development CV macro-F1 (SD)", title="Validation model comparison", xlim=(0, 1))
    fig.tight_layout(); fig.savefig(output / "validation_comparison.png"); plt.close(fig)
    manifest = {"data_sha256": digest(resolved), "assignments_sha256": digest(assignments),
                "champion": champion, "comparator": comparator,
                "selection_metric": "development repeated-CV macro-F1", "folds": len(folds),
                "best_parameters": search.best_params_, "optimized_cv_macro_f1": search.best_score_,
                "original_cv_macro_f1": float(summary.set_index("model_key").loc[champion, "mean_validation_macro_f1"]),
                "search_selection_bias": "Best CV score is optimistic because the same folds selected hyperparameters.",
                "test_status": "Already disclosed in original team notebooks; QA rerun is not an untouched test.",
                "python": platform.python_version(),
                "artifact_sha256": {p.name: digest(p) for p in artifacts.glob("*.joblib")}}
    write_json(output / "locked_manifest.json", manifest)
    return manifest


def evaluate(data: Path, assignments: Path, development: Path, artifacts: Path, output: Path) -> dict:
    manifest = json.loads((development / "locked_manifest.json").read_text())
    _, _, partition, resolved = _load_development_data(data, assignments)
    if digest(resolved) != manifest["data_sha256"] or digest(assignments) != manifest["assignments_sha256"]:
        raise ValueError("Dataset or assignments changed after development decisions were locked.")
    for name, expected in manifest["artifact_sha256"].items():
        if digest(artifacts / name) != expected:
            raise ValueError("Fitted model changed after locking: " + name)
    if output.exists():
        raise FileExistsError("Evaluation directory exists. Reuse its predictions; do not repeatedly rescore the test set.")
    cleaned = clean_raw_data(load_raw_data(resolved))
    test = cleaned.loc[partition.loc[partition.split.eq("test"), "row_index"].astype(int)]
    X, y = test.drop(columns=[TARGET_COLUMN]), test[TARGET_COLUMN].astype(int)
    models = {spec.key: joblib.load(artifacts / f"{spec.key}.joblib") for spec in build_model_specs()}
    models["optimized_champion"] = joblib.load(artifacts / "optimized_champion.joblib")
    # Compute the batch before creating output so a prediction failure remains retryable.
    prediction = {key: model.predict(X) for key, model in models.items()}
    probability = {key: model.predict_proba(X)[:, list(model.classes_).index(1)] for key, model in models.items()}
    output.mkdir(parents=True)
    paired = pd.DataFrame({ID_COLUMN: test[ID_COLUMN].to_numpy(), "actual": y.to_numpy()})
    metrics = []
    fig, ax = plt.subplots()
    for key in models:
        paired[key + "_prediction"] = prediction[key]
        paired[key + "_probability_class_1"] = probability[key]
        metrics.append({"model_key": key, **metric_values(y, prediction[key], probability[key])})
        write_json(output / f"{key}_classification_report.json", classification_report(
            y, prediction[key], labels=[0, 1], target_names=["Reliable (>=60%)", "Low reliability (<60%)"],
            output_dict=True, zero_division=0))
        matrix = confusion_matrix(y, prediction[key], labels=[0, 1])
        pd.DataFrame(matrix, index=["actual_0", "actual_1"], columns=["predicted_0", "predicted_1"]).to_csv(output / f"{key}_confusion_matrix.csv")
        cm_fig, cm_ax = plt.subplots()
        ConfusionMatrixDisplay(matrix, display_labels=["Reliable", "Low reliability"]).plot(ax=cm_ax, colorbar=False)
        cm_ax.set_title(key); cm_fig.tight_layout(); cm_fig.savefig(output / f"{key}_confusion_matrix.png"); plt.close(cm_fig)
        fpr, tpr, _ = roc_curve(y, probability[key])
        ax.plot(fpr, tpr, label=key)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="QA held-out ROC curves")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(output / "common_roc.png"); plt.close(fig)
    paired.to_csv(output / "paired_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "held_out_metrics.csv", index=False)
    write_json(output / "uncertainty.json", {key: uncertainty(y, prediction[key]) for key in models})
    # Rural/urban context is descriptive. It is not a demographic fairness certification.
    from .features import ReliabilityFeatureEngineer
    groups = ReliabilityFeatureEngineer().fit_transform(X)["dominant_population_group"].fillna("unknown")
    subgroup_rows = []
    for group in groups.unique():
        mask = groups.eq(group).to_numpy()
        actual = y.to_numpy()[mask]
        predicted = prediction["optimized_champion"][mask]
        subgroup_rows.append({"group": group, "cases": int(mask.sum()), "positive_cases": int((actual == 1).sum()),
                              "accuracy": accuracy_score(actual, predicted),
                              "recall_class_1": recall_score(actual, predicted, zero_division=0) if (actual == 1).any() else None,
                              "false_negatives": int(((actual == 1) & (predicted == 0)).sum()),
                              "false_positives": int(((actual == 0) & (predicted == 1)).sum())})
    pd.DataFrame(subgroup_rows).to_csv(output / "population_context_errors.csv", index=False)
    result = paired_test(y.to_numpy(), prediction["optimized_champion"], prediction[manifest["comparator"]])
    result.update(champion=manifest["champion"], comparator=manifest["comparator"],
                  test_status=manifest["test_status"], cases=len(y))
    write_json(output / "mcnemar.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["develop", "evaluate"])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS_PATH)
    parser.add_argument("--development", type=Path, default=Path("reports/qa/development"))
    parser.add_argument("--artifacts", type=Path, default=Path("models/qa"))
    parser.add_argument("--evaluation", type=Path, default=Path("reports/qa/evaluation"))
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.stage == "develop":
        result = develop(args.data, args.assignments, args.development, args.artifacts, args.jobs)
    else:
        result = evaluate(args.data, args.assignments, args.development, args.artifacts, args.evaluation)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
