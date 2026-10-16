from pathlib import Path

import polars as pl
import pytest

from alberta_rent_watch.statcan import load_population, load_unemployment

RAW_DIR = Path(__file__).parents[1] / "data" / "raw"


def test_loads_actual_economic_exports():
    population = load_population(RAW_DIR)
    unemployment = load_unemployment(RAW_DIR)

    assert population.schema == {"year": pl.Int64, "population": pl.Int64}
    assert population["year"].to_list() == list(range(2001, 2026))
    assert population.row(0) == (2001, 963062)
    assert population.row(-1) == (2025, 1692385)
    assert unemployment.schema == {"year": pl.Int64, "unemployment_rate": pl.Float64}
    assert unemployment["year"].to_list() == list(range(2011, 2026))
    assert unemployment.row(-1) == (2025, 7.7)


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        ("17-10-0148-01", "00-00-0000-00", "expected table"),
        ("All ages", "15 years and over", "Age group"),
        ('"Persons"', '"Percent"', "expected units"),
        ('"2002"', '"2001"', "duplicate years"),
        ('"963,062"', '"-1"', "Invalid population"),
        ('"963,062"', '"nan"', "Non-finite"),
        ("Edmonton (CMA), Alberta", "Calgary (CMA), Alberta", "exactly one"),
    ],
)
def test_rejects_invalid_population_exports(tmp_path, original, replacement, message):
    source = (RAW_DIR / "statcan_population.csv").read_text(encoding="utf-8-sig")
    (tmp_path / "statcan_population.csv").write_text(
        source.replace(original, replacement), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=message):
        load_population(tmp_path)


def test_preserves_suppressed_unemployment_as_null(tmp_path):
    source = (RAW_DIR / "statcan_unemployment.csv").read_text(encoding="utf-8-sig")
    (tmp_path / "statcan_unemployment.csv").write_text(
        source.replace('"5.5"', '"x"'), encoding="utf-8"
    )
    assert load_unemployment(tmp_path).row(0) == (2011, None)
