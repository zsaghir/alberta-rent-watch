"""Load locally downloaded Statistics Canada rental-market CSV exports."""

from __future__ import annotations

import csv
from pathlib import Path

import polars as pl

RENT_CSV = "cmhc_average_rent_2000_2025.csv"
VACANCY_CSV = "cmhc_vacancy_rate_2000_2025.csv"

RENT_TABLE = "34-10-0133-01"
VACANCY_TABLE = "34-10-0130-01"
TARGET_CITY = "Edmonton, Alberta"
TARGET_STRUCTURE = "Row and apartment structures of three units and over"
TARGET_UNIT = "Two bedroom units"
FIRST_YEAR = 2000
LAST_YEAR = 2025


def _read_rows(path: Path, table: str, source_name: str) -> list[list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"{source_name} CSV not found: {path}")

    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.reader(source))

    if not any(table in cell for row in rows for cell in row):
        raise ValueError(f"{source_name} must be Statistics Canada table {table}")
    return rows


def _find_row(rows: list[list[str]], first_cell: str, source_name: str) -> list[str]:
    matches = [row for row in rows if row and row[0].strip() == first_cell]
    if len(matches) != 1:
        raise ValueError(f"{source_name} must contain exactly one {first_cell!r} row")
    return matches[0]


def _year_columns(row: list[str], source_name: str) -> list[tuple[int, int]]:
    columns: list[tuple[int, int]] = []
    for index, value in enumerate(row[1:], start=1):
        value = value.strip()
        if value.isdigit():
            year = int(value)
            if FIRST_YEAR <= year <= LAST_YEAR:
                columns.append((index, year))

    years = [year for _, year in columns]
    if not years:
        raise ValueError(f"{source_name} contains no years from 2000 through 2025")
    if len(years) != len(set(years)):
        raise ValueError(f"{source_name} contains duplicate year columns")
    return columns


def _number(value: str, source_name: str, year: int) -> float | None:
    value = value.strip().replace(",", "")
    if value in {"", ".."}:
        return None
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(
            f"{source_name} has a non-numeric value for {year}: {value!r}"
        ) from error


def _values_by_year(
    value_row: list[str],
    columns: list[tuple[int, int]],
    source_name: str,
) -> dict[int, float | None]:
    values: dict[int, float | None] = {}
    for index, year in columns:
        if index >= len(value_row):
            raise ValueError(f"{source_name} has no value for {year}")
        values[year] = _number(value_row[index], source_name, year)
    return values


def _load_rent(path: Path) -> dict[int, float | None]:
    source_name = "Rent data"
    rows = _read_rows(path, RENT_TABLE, source_name)
    geography = _find_row(rows, "Geography", source_name)
    unit = _find_row(rows, "Type of unit", source_name)
    year_row = _find_row(rows, "Type of structure", source_name)
    value_row = _find_row(rows, TARGET_STRUCTURE, source_name)

    if len(geography) < 2 or geography[1].strip() != TARGET_CITY:
        raise ValueError(f"{source_name} must be filtered to {TARGET_CITY}")
    if len(unit) < 2 or unit[1].strip() != TARGET_UNIT:
        raise ValueError(f"{source_name} must be filtered to {TARGET_UNIT}")

    return _values_by_year(value_row, _year_columns(year_row, source_name), source_name)


def _load_vacancy(path: Path) -> dict[int, float | None]:
    source_name = "Vacancy data"
    rows = _read_rows(path, VACANCY_TABLE, source_name)
    year_row = _find_row(rows, "Geography", source_name)
    value_row = _find_row(rows, TARGET_CITY, source_name)
    return _values_by_year(value_row, _year_columns(year_row, source_name), source_name)


def load_cmhc_data(raw_dir: Path) -> pl.DataFrame:
    """Load and join Edmonton rent and vacancy exports for 2000–2025."""
    raw_dir = Path(raw_dir)
    rent = _load_rent(raw_dir / RENT_CSV)
    vacancy = _load_vacancy(raw_dir / VACANCY_CSV)

    rent_years = set(rent)
    vacancy_years = set(vacancy)
    if rent_years != vacancy_years:
        raise ValueError(
            "Rent and vacancy years do not match. "
            f"Missing from rent: {sorted(vacancy_years - rent_years)}; "
            f"missing from vacancy: {sorted(rent_years - vacancy_years)}"
        )

    return pl.DataFrame(
        {
            "year": sorted(rent_years),
            "city": [TARGET_CITY] * len(rent_years),
            "average_rent": [rent[year] for year in sorted(rent_years)],
            "vacancy_rate": [vacancy[year] for year in sorted(rent_years)],
        },
        schema={
            "year": pl.Int64,
            "city": pl.String,
            "average_rent": pl.Float64,
            "vacancy_rate": pl.Float64,
        },
    )
