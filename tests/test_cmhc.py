import csv
from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from alberta_rent_watch.cmhc import (
    RENT_ARCHIVE,
    RENT_CSV,
    TARGET_STRUCTURE,
    TARGET_UNIT,
    VACANCY_ARCHIVE,
    VACANCY_CSV,
    load_cmhc_data,
)

RENT_COLUMNS = [
    "REF_DATE",
    "GEO",
    "DGUID",
    "Type of structure",
    "Type of unit",
    "VALUE",
    "STATUS",
]
VACANCY_COLUMNS = ["REF_DATE", "GEO", "DGUID", "VALUE", "STATUS"]


def _write_csv_zip(
    raw_dir: Path,
    archive_name: str,
    csv_name: str,
    columns: list[str],
    rows: list[list[object]],
) -> None:
    csv_buffer = StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(columns)
    writer.writerows(rows)
    with ZipFile(raw_dir / archive_name, "w") as archive:
        archive.writestr(csv_name, csv_buffer.getvalue())


def _write_sources(
    raw_dir: Path,
    rent_rows: list[list[object]],
    vacancy_rows: list[list[object]],
    rent_columns: list[str] = RENT_COLUMNS,
) -> None:
    _write_csv_zip(raw_dir, RENT_ARCHIVE, RENT_CSV, rent_columns, rent_rows)
    _write_csv_zip(
        raw_dir,
        VACANCY_ARCHIVE,
        VACANCY_CSV,
        VACANCY_COLUMNS,
        vacancy_rows,
    )


def _rent_row(
    city: str,
    value: object,
    *,
    year: int = 2024,
    structure: str = TARGET_STRUCTURE,
    unit: str = TARGET_UNIT,
    status: object = None,
) -> list[object]:
    return [year, city, f"dguid-{city}", structure, unit, value, status]


def _vacancy_row(
    city: str, value: object, *, year: int = 2024, status: object = None
) -> list[object]:
    return [year, city, f"dguid-{city}", value, status]


def test_load_filters_to_target_cities_and_categories(tmp_path: Path) -> None:
    rent_rows = [
        _rent_row("Calgary, Alberta", 1876),
        _rent_row("Edmonton, Alberta", 1536),
        _rent_row("Calgary, Alberta", 1500, unit="One bedroom units"),
        _rent_row("Toronto, Ontario", 2000),
    ]
    vacancy_rows = [
        _vacancy_row("Calgary, Alberta", 4.6),
        _vacancy_row("Edmonton, Alberta", 3.0),
        _vacancy_row("Toronto, Ontario", 2.0),
    ]
    _write_sources(tmp_path, rent_rows, vacancy_rows)

    result = load_cmhc_data(tmp_path)

    assert result.select("year", "city", "average_rent", "vacancy_rate").to_dicts() == [
        {
            "year": 2024,
            "city": "Calgary, Alberta",
            "average_rent": 1876.0,
            "vacancy_rate": 4.6,
        },
        {
            "year": 2024,
            "city": "Edmonton, Alberta",
            "average_rent": 1536.0,
            "vacancy_rate": 3.0,
        },
    ]


def test_load_rejects_missing_required_column(tmp_path: Path) -> None:
    rent_columns = [column for column in RENT_COLUMNS if column != "Type of unit"]
    rent_row = _rent_row("Calgary, Alberta", 1876)
    rent_row.pop(RENT_COLUMNS.index("Type of unit"))
    _write_sources(
        tmp_path,
        [rent_row],
        [_vacancy_row("Calgary, Alberta", 4.6)],
        rent_columns,
    )

    with pytest.raises(ValueError, match="Type of unit"):
        load_cmhc_data(tmp_path)


def test_load_rejects_duplicate_city_year(tmp_path: Path) -> None:
    rent_row = _rent_row("Calgary, Alberta", 1876)
    _write_sources(
        tmp_path,
        [rent_row, rent_row],
        [_vacancy_row("Calgary, Alberta", 4.6)],
    )

    with pytest.raises(ValueError, match="duplicate city-year"):
        load_cmhc_data(tmp_path)


def test_load_preserves_suppressed_value_as_null(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        [_rent_row("Calgary, Alberta", None, status="F")],
        [_vacancy_row("Calgary, Alberta", 4.6)],
    )

    result = load_cmhc_data(tmp_path)

    assert result["average_rent"].to_list() == [None]
    assert result["rent_status"].to_list() == ["F"]


def test_load_rejects_mismatched_city_year_keys(tmp_path: Path) -> None:
    _write_sources(
        tmp_path,
        [
            _rent_row("Calgary, Alberta", 1876),
            _rent_row("Edmonton, Alberta", 1536),
        ],
        [_vacancy_row("Calgary, Alberta", 4.6)],
    )

    with pytest.raises(ValueError, match="city-year keys do not match"):
        load_cmhc_data(tmp_path)
