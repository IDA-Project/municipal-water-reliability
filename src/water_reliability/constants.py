"""Shared, pre-declared constants for Questions 1 through 2.4."""

from pathlib import Path

RAW_FILE_NAME = "NIWIS_Water Supply Reliability - Population_26-Sep-2026.csv"
DEFAULT_DATA_PATH = Path("data/raw") / RAW_FILE_NAME
NATIONAL_FILE_NAME = "Water Supply Reliability  - Population - National[Jul-2026] (1).csv"
DEFAULT_NATIONAL_DATA_PATH = Path("data/raw") / NATIONAL_FILE_NAME
DEFAULT_REPORT_DIR = Path("reports/q2")
DEFAULT_ARTIFACT_DIR = Path("models")

RANDOM_STATE = 42
TARGET_THRESHOLD = 60.0
TARGET_COLUMN = "low_reliability"
ID_COLUMN = "WSA Name"
TARGET_SOURCE_COLUMN = "% Population Reliable Supply"

COLUMN_DESCRIPTIONS = {
    ID_COLUMN: "Name of the Water Services Authority (WSA).",
    "Population": "Total population represented by the WSA record.",
    "Population with access to water": "Population with access to water infrastructure.",
    "Population Reliable Supply": "Population recorded as receiving reliable water supply.",
    TARGET_SOURCE_COLUMN: "Population receiving reliable supply, as a percentage.",
    "Urban Population": "Urban population represented by the WSA record.",
    "Urban Population with access to water": "Urban population with access to water infrastructure.",
    "Urban Population Reliable Supply": "Urban population recorded as receiving reliable supply.",
    "% Urban Population Reliable Supply": "Urban population receiving reliable supply, as a percentage.",
    "Rural Population": "Rural population represented by the WSA record.",
    "Rural Population with access to water": "Rural population with access to water infrastructure.",
    "Rural Population Reliable Supply": "Rural population recorded as receiving reliable supply.",
    "% Rural Population Reliable Supply": "Rural population receiving reliable supply, as a percentage.",
}

COUNT_COLUMNS = (
    "Population",
    "Population with access to water",
    "Population Reliable Supply",
    "Urban Population",
    "Urban Population with access to water",
    "Urban Population Reliable Supply",
    "Rural Population",
    "Rural Population with access to water",
    "Rural Population Reliable Supply",
)

PERCENT_COLUMNS = (
    TARGET_SOURCE_COLUMN,
    "% Urban Population Reliable Supply",
    "% Rural Population Reliable Supply",
)

RELIABILITY_COLUMNS = (
    "Population Reliable Supply",
    TARGET_SOURCE_COLUMN,
    "Urban Population Reliable Supply",
    "% Urban Population Reliable Supply",
    "Rural Population Reliable Supply",
    "% Rural Population Reliable Supply",
)

SAFE_RAW_FEATURES = (
    "Population",
    "Population with access to water",
    "Urban Population",
    "Urban Population with access to water",
    "Rural Population",
    "Rural Population with access to water",
)

ENGINEERED_NUMERIC_FEATURES = (
    "population",
    "population_with_access",
    "urban_population",
    "urban_population_with_access",
    "rural_population",
    "rural_population_with_access",
    "access_rate_pct",
    "urban_access_rate_pct",
    "rural_access_rate_pct",
    "urban_population_share_pct",
    "rural_population_share_pct",
)

ENGINEERED_CATEGORICAL_FEATURES = ("dominant_population_group",)

# The teammate's Q2.3 analysis reconstructed these groups from the original row
# order. This is retained only for a sensitivity analysis and is never a model
# feature. Reordering or filtering the raw extract invalidates this mapping.
PROVINCE_ROW_COUNTS = {
    "EC": 15,
    "FS": 18,
    "GP": 9,
    "KZN": 14,
    "LP": 10,
    "MP": 17,
    "NW": 10,
    "NC": 26,
    "WC": 25,
}
