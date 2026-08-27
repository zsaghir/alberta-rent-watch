from pathlib import Path

import pytest

from alberta_rent_watch.cmhc import load_cmhc_data
from alberta_rent_watch.summary import derive_current_rent_summary

RAW_DIR = Path(__file__).parents[1] / "data/raw"


def test_derives_current_rent_summary() -> None:
    summary = derive_current_rent_summary(load_cmhc_data(RAW_DIR))
    assert summary == {
        "latest_year": 2025,
        "latest_rent": 1598.0,
        "change_from_prior_year": 69.0,
        "change_since_2020": 328.0,
        "change_since_2000": 995.0,
    }


def test_requires_comparison_year() -> None:
    data = load_cmhc_data(RAW_DIR)
    data = data.filter(data["year"] != 2020)
    with pytest.raises(KeyError, match="2020"):
        derive_current_rent_summary(data)
