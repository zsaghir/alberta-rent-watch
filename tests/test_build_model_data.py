from pathlib import Path

import pandas as pd
import pytest

from data.build_model_data import MODEL_COLUMNS, add_lag_features, build_model_data

PROJECT_ROOT = Path(__file__).parents[1]
MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "edmonton_2br_master.csv"
MODEL_DATA_PATH = PROJECT_ROOT / "data" / "model" / "edmonton_2br_model.csv"


def make_master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2002, 2000, 2001],
            "average_rent": [710.0, 603.0, 657.0],
            "vacancy_rate": [1.6, 1.3, 0.9],
        }
    )


def test_adds_prior_year_values_in_chronological_order() -> None:
    result = add_lag_features(make_master())

    assert result.columns.tolist() == MODEL_COLUMNS
    assert result.to_dict("records") == [
        {
            "year": 2001,
            "average_rent": 657.0,
            "vacancy_rate": 0.9,
            "rent_lag1": 603.0,
            "vacancy_lag1": 1.3,
        },
        {
            "year": 2002,
            "average_rent": 710.0,
            "vacancy_rate": 1.6,
            "rent_lag1": 657.0,
            "vacancy_lag1": 0.9,
        },
    ]


def test_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="Missing required columns.*vacancy_rate"):
        add_lag_features(make_master().drop(columns="vacancy_rate"))


@pytest.mark.parametrize("years", [[2000, 2001, 2001], [2000, 2001, 2003]])
def test_rejects_duplicate_or_missing_years(years: list[int]) -> None:
    master = make_master().assign(year=years)

    with pytest.raises(ValueError, match="unique and consecutive"):
        add_lag_features(master)


def test_rebuilds_committed_model_dataset(tmp_path: Path) -> None:
    output_path = tmp_path / "model.csv"

    result = build_model_data(MASTER_PATH, output_path)

    assert len(result) == 25
    assert result["year"].tolist() == list(range(2001, 2026))
    assert output_path.read_text() == MODEL_DATA_PATH.read_text()
