"""Q2.4 preprocessing and training-only imbalance treatment."""

from __future__ import annotations

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .constants import (
    ENGINEERED_CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    RANDOM_STATE,
)
from .features import ReliabilityFeatureEngineer


def build_column_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", RobustScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(ENGINEERED_NUMERIC_FEATURES)),
            ("categorical", categorical, list(ENGINEERED_CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_preprocessing_pipeline() -> Pipeline:
    """Pipeline fitted only on a development set or training fold."""
    return Pipeline(
        steps=[
            ("feature_engineering", ReliabilityFeatureEngineer()),
            ("preprocess", build_column_preprocessor()),
        ]
    )


def build_training_preparation_pipeline() -> ImbalancedPipeline:
    """Preparation through resampling, for training data only.

    Q3 should append an estimator after ``balance`` so cross-validation calls
    resampling only within each training fold. Validation and test cases must
    never be passed to ``fit_resample``.
    """
    return ImbalancedPipeline(
        steps=[
            ("feature_engineering", ReliabilityFeatureEngineer()),
            ("preprocess", build_column_preprocessor()),
            ("balance", RandomOverSampler(random_state=RANDOM_STATE)),
        ]
    )

