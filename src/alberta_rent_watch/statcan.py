"""Load Edmonton population and unemployment from Statistics Canada CSVs."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import polars as pl


def _read_rows(path: Path, table_id: str) -> list[list[str]]:
    """Read CSV rows and validate the table identity and frequency."""
    with path.open(mode="r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.reader(source))

    cells = {cell.strip() for row in rows for cell in row}

    if f"Table: {table_id}" not in cells:
        raise ValueError(f"{path.name}: expected table {table_id}")

    if not any(" ".join(cell.split()) == "Frequency: Annual" for cell in cells):
        raise ValueError(f"{path.name}: expected annual data")

    return rows


def _clean_label(value: str) -> str:
    return re.sub(r"(?:\s+\d+)+$", "", value.strip())


def _find_row(rows: list[list[str]], label: str) -> list[str]:
    matches = [row for row in rows if row and _clean_label(row[0]) == label]

    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {label!r} row")

    return matches[0]


def _check_filter(
    rows: list[list[str]],
    label: str,
    expected: str,
) -> None:
    row = _find_row(rows, label)

    if len(row) < 2 or _clean_label(row[1]) != expected:
        raise ValueError(f"{label}: expected {expected!r}")


def _parse_number(value: str, year: int) -> float | None:
    cleaned = value.strip().replace(",", "")

    if cleaned in {"", "..", "x"}:
        return None

    try:
        number = float(cleaned)
    except ValueError as error:
        raise ValueError(f"Invalid value for {year}: {value!r}") from error

    if not math.isfinite(number):
        raise ValueError(f"Non-finite value for {year}: {value!r}")

    return number


def _load_series(
    path: Path,
    *,
    table_id: str,
    geography: str,
    filters: dict[str, str],
    unit: str,
    column: str,
    first_year: int,
    last_year: int,
) -> pl.DataFrame:
    """Validate a wide annual export and return one row per year."""
    rows = _read_rows(path, table_id)

    for label, expected in filters.items():
        _check_filter(rows, label, expected)

    year_row = _find_row(rows, "Geography")
    value_row = _find_row(rows, geography)

    # These exports place the units directly below the year header.
    unit_position = rows.index(year_row) + 1
    unit_row = rows[unit_position] if unit_position < len(rows) else []
    if len(unit_row) < 2 or unit_row[0].strip() or unit_row[1].strip() != unit:
        raise ValueError(f"{path.name}: expected units {unit!r}")

    years = []
    values = []
    for position, cell in enumerate(year_row[1:], start=1):
        text = cell.strip()
        if not text:
            continue
        if not re.fullmatch(r"\d{4}", text):
            raise ValueError(f"Invalid year header: {cell!r}")

        year = int(text)
        if position >= len(value_row):
            raise ValueError(f"{path.name}: missing cell for {year}")

        years.append(year)
        values.append(_parse_number(value_row[position], year))

    if len(years) != len(set(years)):
        raise ValueError(f"{path.name}: duplicate years")
    if sorted(years) != list(range(first_year, last_year + 1)):
        raise ValueError(f"{path.name}: expected coverage {first_year}–{last_year}")

    for year, value in zip(years, values):
        if value is None:
            continue
        if column == "population":
            if value <= 0 or not value.is_integer():
                raise ValueError(f"Invalid population for {year}: {value}")
        elif not 0 <= value <= 100:
            raise ValueError(f"Invalid unemployment rate for {year}: {value}")

    if column == "population":
        values = [None if value is None else int(value) for value in values]
        dtype = pl.Int64
    else:
        dtype = pl.Float64

    return pl.DataFrame(
        {"year": years, column: values},
        schema={"year": pl.Int64, column: dtype},
    ).sort("year")


def load_population(raw_dir: Path) -> pl.DataFrame:
    """Load annual Edmonton CMA population estimates in persons."""
    return _load_series(
        Path(raw_dir) / "statcan_population.csv",
        table_id="17-10-0148-01",
        geography="Edmonton (CMA), Alberta",
        filters={
            "Gender": "Total - gender",
            "Age group": "All ages",
        },
        unit="Persons",
        column="population",
        first_year=2001,
        last_year=2025,
    )


def load_unemployment(raw_dir: Path) -> pl.DataFrame:
    """Load annual Edmonton CMA unemployment rates in percent."""
    return _load_series(
        Path(raw_dir) / "statcan_unemployment.csv",
        table_id="14-10-0461-01",
        geography="Edmonton, Alberta",
        filters={
            "Labour force characteristics": "Unemployment rate",
            "Gender": "Total - Gender",
            "Age group": "15 years and over",
        },
        unit="Percent",
        column="unemployment_rate",
        first_year=2011,
        last_year=2025,
    )
