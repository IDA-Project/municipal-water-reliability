"""Loading, schema validation, cleaning, and target construction."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .constants import (
    COLUMN_DESCRIPTIONS,
    COUNT_COLUMNS,
    ID_COLUMN,
    PERCENT_COLUMNS,
    TARGET_COLUMN,
    TARGET_SOURCE_COLUMN,
    TARGET_THRESHOLD,
)


def resolve_data_path(requested_path: str | Path) -> Path:
    """Return the requested CSV path or raise an actionable error."""
    path = Path(requested_path)
    if path.is_file():
        return path

    candidates = sorted(path.parent.glob("*NIWIS*Water*Supply*Reliability*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(str(item) for item in candidates)
        raise FileNotFoundError(
            f"The requested dataset was not found and multiple NIWIS CSVs exist: {names}. "
            "Pass the intended file with --data."
        )
    raise FileNotFoundError(
        f"Dataset not found at '{path}'. Copy the unchanged NIWIS export into "
        "data/raw/ or pass its exact path with --data."
    )


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the unchanged CSV without learning or filling any values."""
    resolved = resolve_data_path(path)
    return pd.read_csv(resolved)


def load_national_summary(path: str | Path) -> pd.Series:
    """Load the nine-row national control total file as a named Series."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"National summary dataset not found at '{resolved}'.")
    raw = pd.read_csv(resolved)
    if raw.shape[1] != 2:
        raise ValueError("The national summary must contain exactly two columns.")
    categories = (
        raw.iloc[:, 0]
        .astype("string")
        .str.strip()
        .str.lstrip("'")
        .str.replace("Reliability Supply", "Reliable Supply", regex=False)
    )
    values = _parse_numeric(raw.iloc[:, 1])
    return pd.Series(values.to_numpy(), index=categories, name="national_value", dtype="float64")


def _normalise_column_name(value: object) -> str:
    text = str(value).replace("\ufeff", "").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _parse_numeric(series: pd.Series, percentage: bool = False) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace(
        {"": pd.NA, "-": pd.NA, "--": pd.NA, "n/a": pd.NA, "N/A": pd.NA, "null": pd.NA}
    )
    cleaned = cleaned.str.replace("\u00a0", "", regex=False)
    cleaned = cleaned.str.replace(r"[\s,']", "", regex=True)
    if percentage:
        cleaned = cleaned.str.replace("%", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def validate_required_schema(frame: pd.DataFrame) -> None:
    missing = sorted(set(COLUMN_DESCRIPTIONS) - set(frame.columns))
    if missing:
        raise ValueError(f"The NIWIS extract is missing required columns: {missing}")


def clean_raw_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse the NIWIS export and construct the pre-declared binary target.

    Cleaning is deterministic and does not learn statistics from the data.
    Imputation and scaling are deliberately deferred to the training pipeline.
    """
    frame = raw.copy()
    frame.columns = [_normalise_column_name(column) for column in frame.columns]
    unnamed = [column for column in frame.columns if column.lower().startswith("unnamed")]
    if unnamed:
        frame = frame.drop(columns=unnamed)

    validate_required_schema(frame)
    frame[ID_COLUMN] = frame[ID_COLUMN].astype("string").str.strip().str.lstrip("'")
    frame[ID_COLUMN] = frame[ID_COLUMN].replace({"": pd.NA})

    for column in COUNT_COLUMNS:
        frame[column] = _parse_numeric(frame[column])
    for column in PERCENT_COLUMNS:
        frame[column] = _parse_numeric(frame[column], percentage=True)

    target_source = frame[TARGET_SOURCE_COLUMN]
    frame[TARGET_COLUMN] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    known = target_source.notna()
    frame.loc[known, TARGET_COLUMN] = (
        target_source.loc[known] < TARGET_THRESHOLD
    ).astype("int64")
    return frame


def safe_percentage(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_values = pd.to_numeric(numerator, errors="coerce").astype("float64")
    denominator_values = pd.to_numeric(denominator, errors="coerce").astype("float64")
    valid = denominator_values.gt(0)
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    result.loc[valid] = numerator_values.loc[valid] / denominator_values.loc[valid] * 100.0
    return result
