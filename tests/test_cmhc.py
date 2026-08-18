import csv
from collections.abc import Sequence
from pathlib import Path

import pytest

from alberta_rent_watch.cmhc import (
    RENT_CSV,
    TARGET_CITY,
    TARGET_STRUCTURE,
    TARGET_UNIT,
    VACANCY_CSV,
    load_cmhc_data,
)


def _write_rows(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        csv.writer(destination).writerows(rows)


def _write_rent(
    raw_dir: Path,
    years: Sequence[int] = (2023, 2024),
    values: Sequence[object] = (1398, 1529),
    *,
    city: str = TARGET_CITY,
) -> None:
    _write_rows(
        raw_dir / RENT_CSV,
        [
            ["Canada Mortgage and Housing Corporation, average rents"],
            ["Table: 34-10-0133-01"],
            [],
            ["Geography", city, *([""] * (len(years) - 1))],
            ["Type of unit", TARGET_UNIT, *([""] * (len(years) - 1))],
            ["Type of structure", *years],
            ["", "Dollars", *([""] * (len(years) - 1))],
            [TARGET_STRUCTURE, *values],
            [],
            ["Footnotes:"],
        ],
    )


def _write_vacancy(
    raw_dir: Path,
    years: Sequence[int] = (2023, 2024),
    values: Sequence[object] = (2.3, 3.0),
) -> None:
    _write_rows(
        raw_dir / VACANCY_CSV,
        [
            ["Canada Mortgage and Housing Corporation, vacancy rates"],
            ["Table: 34-10-0130-01"],
            [],
            ["Geography", *years],
            ["", "Rate", *([""] * (len(years) - 1))],
            [TARGET_CITY, *values],
            [],
            ["Footnotes:"],
        ],
    )


def test_loads_wide_local_exports_as_long_data(tmp_path: Path) -> None:
    _write_rent(tmp_path)
    _write_vacancy(tmp_path)

    result = load_cmhc_data(tmp_path)

    assert result.to_dicts() == [
        {
            "year": 2023,
            "city": TARGET_CITY,
            "average_rent": 1398.0,
            "vacancy_rate": 2.3,
        },
        {
            "year": 2024,
            "city": TARGET_CITY,
            "average_rent": 1529.0,
            "vacancy_rate": 3.0,
        },
    ]


def test_loads_comma_numbers_and_missing_symbols(tmp_path: Path) -> None:
    _write_rent(tmp_path, values=["1,398", ".."])
    _write_vacancy(tmp_path, values=["..", 3.0])

    result = load_cmhc_data(tmp_path)

    assert result["average_rent"].to_list() == [1398.0, None]
    assert result["vacancy_rate"].to_list() == [None, 3.0]


def test_rejects_wrong_table_for_rent(tmp_path: Path) -> None:
    _write_vacancy(tmp_path)
    (tmp_path / VACANCY_CSV).replace(tmp_path / RENT_CSV)
    _write_vacancy(tmp_path)

    with pytest.raises(ValueError, match="34-10-0133-01"):
        load_cmhc_data(tmp_path)


def test_rejects_rent_for_wrong_city(tmp_path: Path) -> None:
    _write_rent(tmp_path, city="Calgary, Alberta")
    _write_vacancy(tmp_path)

    with pytest.raises(ValueError, match="filtered to Edmonton"):
        load_cmhc_data(tmp_path)


def test_rejects_duplicate_year_columns(tmp_path: Path) -> None:
    _write_rent(tmp_path, years=[2024, 2024])
    _write_vacancy(tmp_path)

    with pytest.raises(ValueError, match="duplicate year"):
        load_cmhc_data(tmp_path)


def test_rejects_mismatched_years(tmp_path: Path) -> None:
    _write_rent(tmp_path)
    _write_vacancy(tmp_path, years=[2024, 2025])

    with pytest.raises(ValueError, match="years do not match"):
        load_cmhc_data(tmp_path)
