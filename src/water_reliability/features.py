"""Leakage-safe, deterministic feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .constants import SAFE_RAW_FEATURES
from .data import safe_percentage


class ReliabilityFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create only predictors available independently of the reliability label."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "ReliabilityFeatureEngineer":
        missing = sorted(set(SAFE_RAW_FEATURES) - set(X.columns))
        if missing:
            raise ValueError(f"Cannot engineer features; missing columns: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        result = pd.DataFrame(index=frame.index)
        result["population"] = frame["Population"]
        result["population_with_access"] = frame["Population with access to water"]
        result["urban_population"] = frame["Urban Population"]
        result["urban_population_with_access"] = frame[
            "Urban Population with access to water"
        ]
        result["rural_population"] = frame["Rural Population"]
        result["rural_population_with_access"] = frame[
            "Rural Population with access to water"
        ]
        result["access_rate_pct"] = safe_percentage(
            frame["Population with access to water"], frame["Population"]
        )
        result["urban_access_rate_pct"] = safe_percentage(
            frame["Urban Population with access to water"], frame["Urban Population"]
        )
        result["rural_access_rate_pct"] = safe_percentage(
            frame["Rural Population with access to water"], frame["Rural Population"]
        )
        result["urban_population_share_pct"] = safe_percentage(
            frame["Urban Population"], frame["Population"]
        )
        result["rural_population_share_pct"] = safe_percentage(
            frame["Rural Population"], frame["Population"]
        )

        urban = result["urban_population_share_pct"]
        rural = result["rural_population_share_pct"]
        result["dominant_population_group"] = np.select(
            [urban.gt(rural), rural.gt(urban), urban.notna() & rural.notna()],
            ["urban", "rural", "balanced"],
            default=None,
        )
        return result

