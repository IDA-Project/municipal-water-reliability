"""Fixed split and cross-validation protocol from the teammate's Q2.3 answer."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, train_test_split

from .constants import ID_COLUMN, PROVINCE_ROW_COUNTS, RANDOM_STATE, TARGET_COLUMN


def assign_province_sensitivity_groups(frame: pd.DataFrame) -> pd.Series:
    """Reconstruct the original 144-row province blocks for sensitivity only."""
    expected = sum(PROVINCE_ROW_COUNTS.values())
    if len(frame) != expected:
        raise ValueError(
            f"Province row-order reconstruction requires the unchanged {expected}-row extract; "
            f"received {len(frame)} rows."
        )
    labels: list[str] = []
    for province, count in PROVINCE_ROW_COUNTS.items():
        labels.extend([province] * count)
    return pd.Series(labels, index=frame.index, name="province_sensitivity_group", dtype="string")


def make_holdout_split(frame: pd.DataFrame, test_size: float = 0.20) -> pd.DataFrame:
    eligible = frame.loc[frame[TARGET_COLUMN].notna()].copy()
    if eligible.empty:
        raise ValueError("No rows have a usable target.")
    if eligible[ID_COLUMN].isna().any() or eligible[ID_COLUMN].duplicated().any():
        raise ValueError("WSA identifiers must be present and unique before splitting.")
    if eligible[TARGET_COLUMN].nunique() < 2:
        raise ValueError("A stratified split requires both target classes.")

    _, test_index = train_test_split(
        eligible.index.to_numpy(),
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=eligible[TARGET_COLUMN].astype(int),
    )
    assignment = pd.DataFrame(
        {
            "row_index": eligible.index,
            ID_COLUMN: eligible[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: eligible[TARGET_COLUMN].astype(int).to_numpy(),
            "split": "development",
        }
    ).set_index("row_index")
    assignment.loc[test_index, "split"] = "test"
    return assignment.sort_index().reset_index()


def build_repeated_cv() -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=RANDOM_STATE,
    )


def build_grouped_sensitivity_cv() -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def summarise_repeated_cv(y: pd.Series) -> pd.DataFrame:
    """Return class counts for every repeated-stratified validation fold."""
    y_values = y.astype(int).reset_index(drop=True)
    rows: list[dict[str, int]] = []
    for fold_index, (train_index, validation_index) in enumerate(
        build_repeated_cv().split(np.zeros(len(y_values)), y_values)
    ):
        rows.append(
            {
                "repeat": fold_index // 5 + 1,
                "fold": fold_index % 5 + 1,
                "training_rows": int(len(train_index)),
                "training_positives": int(y_values.iloc[train_index].sum()),
                "validation_rows": int(len(validation_index)),
                "validation_positives": int(y_values.iloc[validation_index].sum()),
            }
        )
    return pd.DataFrame(rows)


def summarise_grouped_sensitivity_cv(
    frame: pd.DataFrame, groups: pd.Series
) -> pd.DataFrame:
    """Summarise five province-held-out folds without fitting a model."""
    y = frame[TARGET_COLUMN].astype(int)
    rows: list[dict[str, object]] = []
    splitter = build_grouped_sensitivity_cv()
    for fold, (_, validation_index) in enumerate(
        splitter.split(frame, y, groups), start=1
    ):
        validation_groups = sorted(groups.iloc[validation_index].unique().tolist())
        rows.append(
            {
                "fold": fold,
                "validation_rows": int(len(validation_index)),
                "validation_positives": int(y.iloc[validation_index].sum()),
                "held_out_provinces": ", ".join(validation_groups),
            }
        )
    return pd.DataFrame(rows)
