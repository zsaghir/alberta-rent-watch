from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from data.build_master import build_master, validate_master

RAW_DIR = Path(__file__).parents[1] / "data" / "raw"


def test_builds_canonical_master_dataset(tmp_path: Path) -> None:
    output_path = tmp_path / "edmonton_2br_master.csv"

    master = build_master(RAW_DIR, output_path)

    assert master.columns == ["year", "average_rent", "vacancy_rate"]
    assert master.schema == {
        "year": pl.Int64,
        "average_rent": pl.Float64,
        "vacancy_rate": pl.Float64,
    }
    assert master.height == 26
    assert master["year"].to_list() == list(range(2000, 2026))
    assert master.row(0, named=True) == {
        "year": 2000,
        "average_rent": 603.0,
        "vacancy_rate": 1.3,
    }
    assert master.row(-1, named=True) == {
        "year": 2025,
        "average_rent": 1598.0,
        "vacancy_rate": 3.8,
    }
    assert pl.read_csv(output_path).equals(master)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("average_rent", None, "missing rent"),
        ("vacancy_rate", None, "missing vacancy"),
        ("average_rent", 0.0, "greater than zero"),
        ("vacancy_rate", -0.1, "between 0 and 100"),
        ("vacancy_rate", 100.1, "between 0 and 100"),
    ],
)
def test_rejects_invalid_master_values(
    tmp_path: Path,
    column: str,
    value: float | None,
    message: str,
) -> None:
    master = build_master(RAW_DIR, tmp_path / "master.csv")
    invalid = master.with_columns(
        pl.when(pl.col("year") == 2000)
        .then(pl.lit(value))
        .otherwise(pl.col(column))
        .alias(column)
    )

    with pytest.raises(ValueError, match=message):
        validate_master(invalid)
