"""The five purposeful Q2.2 visualisations and data-driven interpretations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import TARGET_COLUMN, TARGET_SOURCE_COLUMN, TARGET_THRESHOLD
from .data import safe_percentage


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _annotate_bars(axis: plt.Axes) -> None:
    for patch in axis.patches:
        value = patch.get_height() if patch.get_height() else patch.get_width()
        if np.isfinite(value):
            if patch.get_height():
                axis.annotate(
                    f"{value:.0f}",
                    (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                    ha="center",
                    va="bottom",
                )


def _scatter_with_fit(
    x: pd.Series,
    y: pd.Series,
    x_label: str,
    title: str,
    path: Path,
) -> float:
    points = pd.DataFrame({"x": x, "y": y}).dropna()
    has_variation = points["x"].nunique() > 1 and points["y"].nunique() > 1
    correlation = (
        float(points["x"].corr(points["y"]))
        if len(points) >= 2 and has_variation
        else float("nan")
    )
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(points["x"], points["y"], alpha=0.72, edgecolor="none")
    if len(points) >= 2 and points["x"].nunique() > 1:
        slope, intercept = np.polyfit(points["x"], points["y"], 1)
        x_line = np.linspace(points["x"].min(), points["x"].max(), 100)
        axis.plot(x_line, slope * x_line + intercept, color="#c44e52", linewidth=2)
    axis.axhline(TARGET_THRESHOLD, color="#555555", linestyle="--", linewidth=1, label="60% target threshold")
    axis.set(title=title, xlabel=x_label, ylabel="Population with reliable supply (%)")
    axis.text(0.02, 0.97, f"Pearson r = {correlation:.2f}", transform=axis.transAxes, va="top")
    axis.legend(loc="lower right")
    _save(fig, path)
    return correlation


def create_eda_figures(frame: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, object] = {}

    counts = frame[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(["Reliable (>=60%)", "Low reliability (<60%)"], counts.values, color=["#4c9f70", "#c44e52"])
    axis.set(title="Figure 1: Target class distribution", xlabel="WSA reliability class", ylabel="Number of WSAs")
    _annotate_bars(axis)
    _save(fig, output_dir / "figure_01_target_distribution.png")
    evidence["class_counts"] = {str(index): int(value) for index, value in counts.items()}

    missing = frame.drop(columns=[TARGET_COLUMN]).isna().mean().mul(100).sort_values(ascending=True)
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(missing.index, missing.values, color="#4c78a8")
    axis.set(title="Figure 2: Missing data by variable", xlabel="Missing observations (%)", ylabel="Variable")
    _save(fig, output_dir / "figure_02_missingness.png")
    evidence["largest_missing_percent"] = float(missing.max())
    evidence["largest_missing_column"] = str(missing.idxmax())

    access_rate = safe_percentage(frame["Population with access to water"], frame["Population"])
    evidence["access_reliability_correlation"] = _scatter_with_fit(
        access_rate,
        frame[TARGET_SOURCE_COLUMN],
        "Population with access to water (%)",
        "Figure 3: Water access and reliable supply",
        output_dir / "figure_03_access_vs_reliability.png",
    )

    rural_share = safe_percentage(frame["Rural Population"], frame["Population"])
    evidence["rural_share_reliability_correlation"] = _scatter_with_fit(
        rural_share,
        frame[TARGET_SOURCE_COLUMN],
        "Rural population share (%)",
        "Figure 4: Rural population share and reliable supply",
        output_dir / "figure_04_rural_share_vs_reliability.png",
    )

    means = pd.Series(
        {
            "Overall": frame[TARGET_SOURCE_COLUMN].mean(),
            "Urban": frame["% Urban Population Reliable Supply"].mean(),
            "Rural": frame["% Rural Population Reliable Supply"].mean(),
        }
    )
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(means.index, means.values, color="#4c78a8")
    axis.set(title="Figure 5: Reliable supply by population group", xlabel="Population group", ylabel="Mean reliable supply (%)", ylim=(0, 100))
    for position, value in enumerate(means.values):
        axis.text(position, value + 1, f"{value:.1f}%", ha="center")
    _save(fig, output_dir / "figure_05_population_group_comparison.png")
    evidence["mean_reliable_supply_percent"] = {
        str(index): float(value) for index, value in means.items()
    }
    return evidence


def write_interpretations(evidence: dict[str, object], path: Path) -> None:
    counts = evidence["class_counts"]
    means = evidence["mean_reliable_supply_percent"]
    text = f"""# Q2.2 figure interpretations

1. **Target distribution.** There are {counts['1']} low-reliability WSAs and {counts['0']} reliable WSAs under the pre-declared 60% threshold. This imbalance motivates macro-F1 and training-fold-only resampling.
2. **Missingness.** `{evidence['largest_missing_column']}` has the highest missingness at {evidence['largest_missing_percent']:.1f}%. Missing predictors are retained and imputed inside training folds; missing targets are excluded from modelling.
3. **Water access and reliability.** The Pearson correlation is {evidence['access_reliability_correlation']:.2f}. Access is related to reliability but does not define it; the scatter should be read as association, not causation.
4. **Rural share and reliability.** The Pearson correlation is {evidence['rural_share_reliability_correlation']:.2f}. This supports checking geographic and rural/urban disparities, but does not establish that rural composition causes poor reliability.
5. **Urban/rural subgroup view.** Mean reliable supply is {means['Urban']:.1f}% for the urban measure and {means['Rural']:.1f}% for the rural measure, compared with {means['Overall']:.1f}% overall. Aggregate means can hide WSA-level variation and are not fairness proof.
"""
    path.write_text(text, encoding="utf-8")
