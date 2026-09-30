from __future__ import annotations

from pathlib import Path

import pytest

from water_reliability.audit import audit_dataset
from water_reliability.constants import (
    DEFAULT_DATA_PATH,
    TARGET_COLUMN,
)
from water_reliability.data import clean_raw_data, load_raw_data
from water_reliability.split import (
    assign_province_sensitivity_groups,
    make_holdout_split,
)


pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_DATA_PATH).is_file(),
    reason="The licence-controlled NIWIS source file is not present locally.",
)


def test_supplied_extract_matches_the_team_data_contract() -> None:
    cleaned = clean_raw_data(load_raw_data(DEFAULT_DATA_PATH))
    audit = audit_dataset(cleaned)

    assert len(cleaned) == 144
    assert cleaned["WSA Name"].nunique() == 144
    assert cleaned[TARGET_COLUMN].value_counts().sort_index().to_dict() == {0: 105, 1: 39}
    assert audit.summary["duplicate_rows"] == 0
    assert audit.summary["rows_with_duplicate_wsa_name"] == 0
    assert audit.summary["missing_cells"] == 45


def test_supplied_extract_reproduces_the_q2_3_split_and_province_counts() -> None:
    cleaned = clean_raw_data(load_raw_data(DEFAULT_DATA_PATH))
    assignment = make_holdout_split(cleaned)
    grouped = assignment.groupby(["split", TARGET_COLUMN]).size().to_dict()
    assert grouped == {
        ("development", 0): 84,
        ("development", 1): 31,
        ("test", 0): 21,
        ("test", 1): 8,
    }

    groups = assign_province_sensitivity_groups(cleaned)
    province_counts = cleaned.groupby(groups)[TARGET_COLUMN].sum().astype(int).to_dict()
    assert province_counts == {
        "EC": 6,
        "FS": 5,
        "GP": 0,
        "KZN": 9,
        "LP": 6,
        "MP": 4,
        "NC": 4,
        "NW": 5,
        "WC": 0,
    }

