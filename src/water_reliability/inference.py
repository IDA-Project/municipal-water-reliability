"""Validate raw WSA counts and run the shipped, locked complete pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from importlib.metadata import version

from .constants import SAFE_RAW_FEATURES


def validate_inputs(values: dict) -> pd.DataFrame:
    missing = set(SAFE_RAW_FEATURES) - set(values)
    extra = set(values) - set(SAFE_RAW_FEATURES)
    if missing or extra:
        raise ValueError("Provide exactly the six population and water-access fields.")
    counts = {}
    for name in SAFE_RAW_FEATURES:
        try:
            if isinstance(values[name], (bool, np.bool_)):
                raise ValueError
            number = float(values[name])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be a number.") from error
        if not np.isfinite(number) or number < 0 or not number.is_integer():
            raise ValueError(f"{name} must be a finite, non-negative whole number.")
        counts[name] = number
    total = counts["Population"]
    if total <= 0:
        raise ValueError("Total population must be greater than zero.")
    for population, access in [
        ("Population", "Population with access to water"),
        ("Urban Population", "Urban Population with access to water"),
        ("Rural Population", "Rural Population with access to water"),
    ]:
        if counts[access] > counts[population]:
            raise ValueError(f"{access} cannot exceed {population}.")
    if counts["Urban Population"] + counts["Rural Population"] != total:
        raise ValueError("Urban and rural populations must add up to total population.")
    if counts["Urban Population with access to water"] + counts["Rural Population with access to water"] != counts["Population with access to water"]:
        raise ValueError("Urban and rural water-access counts must add up to the overall water-access count.")
    return pd.DataFrame([counts], columns=list(SAFE_RAW_FEATURES))


def load_pipeline(path: Path):
    """Load only the packaged trusted artifact after checksum/version checks."""
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("The packaged model checksum does not match its metadata.")
    for package, expected in metadata["runtime_versions"].items():
        if version(package) != expected:
            raise ValueError(f"Model requires {package} {expected}; install requirements-qa.txt.")
    model = joblib.load(path)
    if not {"feature_engineering", "preprocess", "model"}.issubset(model.named_steps):
        raise ValueError("A complete raw-input preprocessing and model pipeline is required.")
    if set(model.classes_) != {0, 1}:
        raise ValueError("The model must use binary classes 0 and 1.")
    return model


def predict(model, values: dict) -> tuple[int, float]:
    frame = validate_inputs(values)
    predicted = int(model.predict(frame)[0])
    probabilities = np.asarray(model.predict_proba(frame), dtype=float)
    if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all():
        raise ValueError("The model returned invalid probabilities.")
    if (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("The model returned invalid probabilities.")
    if predicted not in (0, 1):
        raise ValueError("The model returned an invalid class.")
    return predicted, float(probabilities[0, list(model.classes_).index(1)])
