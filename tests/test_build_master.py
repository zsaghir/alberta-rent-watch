from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from data.build_master import build_master, validate_master

RAW_DIR = Path(__file__).parents[1] / "data" / "raw"


def test_builds_canonical_master_dataset(tmp_path: Path) -> None:
    output_path = tmp_path / "edmonton_2br_master.csv"

    master = build_master(RAW_DIR, output_path)

    assert master.columns == [
        "year",
        "average_rent",
        "vacancy_rate",
        "population",
        "unemployment_rate",
    ]
    assert master.schema == {
        "year": pl.Int64,
        "average_rent": pl.Float64,
        "vacancy_rate": pl.Float64,
        "population": pl.Int64,
        "unemployment_rate": pl.Float64,
    }
    assert master.height == 26
    assert master["year"].to_list() == list(range(2000, 2026))
    assert master.row(0, named=True) == {
        "year": 2000,
        "average_rent": 603.0,
        "vacancy_rate": 1.3,
        "population": None,
        "unemployment_rate": None,
    }
    assert master.row(-1, named=True) == {
        "year": 2025,
        "average_rent": 1598.0,
        "vacancy_rate": 3.8,
        "population": 1692385,
        "unemployment_rate": 7.7,
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


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("population", None, "Missing population"),
        ("population", 0, "greater than zero"),
        ("unemployment_rate", None, "Missing unemployment_rate"),
        ("unemployment_rate", -1.0, "between 0 and 100"),
        ("unemployment_rate", 101.0, "between 0 and 100"),
        ("unemployment_rate", float("nan"), "finite"),
        ("average_rent", float("inf"), "finite"),
        ("vacancy_rate", float("nan"), "finite"),
    ],
)
def test_rejects_invalid_covered_values(tmp_path, column, value, message):
    master = build_master(RAW_DIR, tmp_path / "master.csv")
    invalid = master.with_columns(
        pl.when(pl.col("year") == 2025)
        .then(pl.lit(value))
        .otherwise(pl.col(column))
        .alias(column)
    )
    with pytest.raises(ValueError, match=message):
        validate_master(invalid)


def test_preserves_history_and_expected_gaps(tmp_path):
    from alberta_rent_watch.cmhc import load_cmhc_data

    master = build_master(RAW_DIR, tmp_path / "master.csv")
    assert master.select("year", "average_rent", "vacancy_rate").equals(
        load_cmhc_data(RAW_DIR)
        .select("year", "average_rent", "vacancy_rate")
        .sort("year")
    )
    assert master.filter(pl.col("population").is_null())["year"].to_list() == [2000]
    assert master.filter(pl.col("unemployment_rate").is_null())[
        "year"
    ].to_list() == list(range(2000, 2011))


@pytest.mark.parametrize(
    "change", ["types", "order", "duplicate", "missing", "early_value"]
)
def test_rejects_invalid_master_structure(tmp_path, change):
    master = build_master(RAW_DIR, tmp_path / "master.csv")
    if change == "types":
        master = master.with_columns(pl.col("population").cast(pl.Float64))
    elif change == "order":
        master = master.select(list(reversed(master.columns)))
    elif change == "duplicate":
        master = pl.concat([master.head(25), master.head(1)])
    elif change == "missing":
        master = master.head(25)
    else:
        master = master.with_columns(pl.col("population").fill_null(1))
    with pytest.raises(ValueError):
        validate_master(master)


def test_rejects_duplicate_join_keys(tmp_path, monkeypatch):
    import data.build_master as module
    from alberta_rent_watch.statcan import load_population

    population = load_population(RAW_DIR)
    monkeypatch.setattr(
        module, "load_population", lambda _: pl.concat([population, population.head(1)])
    )
    with pytest.raises(pl.exceptions.ComputeError, match="1:1"):
        build_master(RAW_DIR, tmp_path / "master.csv")


def test_rejects_missing_join_observation(tmp_path, monkeypatch):
    import data.build_master as module
    from alberta_rent_watch.statcan import load_population

    population = load_population(RAW_DIR).filter(pl.col("year") != 2020)
    monkeypatch.setattr(module, "load_population", lambda _: population)
    with pytest.raises(ValueError, match="Missing population.*2020"):
        build_master(RAW_DIR, tmp_path / "master.csv")
