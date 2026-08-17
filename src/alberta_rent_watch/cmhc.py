"""Download and load the initial CMHC rental-market dataset."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import httpx
import polars as pl

RENT_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/34100133-eng.zip"
VACANCY_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/34100130-eng.zip"

RENT_ARCHIVE = "34100133-eng.zip"
VACANCY_ARCHIVE = "34100130-eng.zip"
RENT_CSV = "34100133.csv"
VACANCY_CSV = "34100130.csv"

TARGET_CITIES = ("Calgary, Alberta", "Edmonton, Alberta")
TARGET_STRUCTURE = "Row and apartment structures of three units and over"
TARGET_UNIT = "Two bedroom units"
FIRST_SHARED_YEAR = 1992

RENT_REQUIRED_COLUMNS = {
    "REF_DATE",
    "GEO",
    "DGUID",
    "Type of structure",
    "Type of unit",
    "VALUE",
    "STATUS",
}
VACANCY_REQUIRED_COLUMNS = {
    "REF_DATE",
    "GEO",
    "DGUID",
    "VALUE",
    "STATUS",
}


def download_cmhc_data(raw_dir: Path) -> tuple[Path, Path]:
    """Download the CMHC-sourced rent and vacancy archives if absent."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded_paths = []
    for url, filename in (
        (RENT_URL, RENT_ARCHIVE),
        (VACANCY_URL, VACANCY_ARCHIVE),
    ):
        destination = raw_dir / filename
        if not destination.exists():
            response = httpx.get(url, follow_redirects=True, timeout=60.0)
            response.raise_for_status()
            with destination.open("xb") as raw_file:
                raw_file.write(response.content)
        downloaded_paths.append(destination)

    return downloaded_paths[0], downloaded_paths[1]


def _read_csv_from_zip(archive_path: Path, csv_name: str) -> pl.DataFrame:
    try:
        with ZipFile(archive_path) as archive:
            csv_bytes = archive.read(csv_name)
    except (BadZipFile, KeyError) as error:
        raise ValueError(
            f"Could not read {csv_name!r} from {archive_path}"
        ) from error

    return pl.read_csv(BytesIO(csv_bytes), null_values=[""])


def _require_columns(
    frame: pl.DataFrame, required: set[str], source_name: str
) -> None:
    missing = required.difference(frame.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"{source_name} is missing required columns: {missing_names}")


def _require_unique_city_year(frame: pl.DataFrame, source_name: str) -> None:
    duplicate_count = frame.select(["year", "city"]).is_duplicated().sum()
    if duplicate_count:
        raise ValueError(f"{source_name} contains duplicate city-year observations")


def load_cmhc_data(raw_dir: Path) -> pl.DataFrame:
    """Load the Calgary and Edmonton two-bedroom rent and vacancy series."""
    raw_dir = Path(raw_dir)
    rent_source = _read_csv_from_zip(raw_dir / RENT_ARCHIVE, RENT_CSV)
    vacancy_source = _read_csv_from_zip(raw_dir / VACANCY_ARCHIVE, VACANCY_CSV)

    _require_columns(rent_source, RENT_REQUIRED_COLUMNS, "Rent data")
    _require_columns(vacancy_source, VACANCY_REQUIRED_COLUMNS, "Vacancy data")

    rent = (
        rent_source.filter(
            pl.col("GEO").is_in(TARGET_CITIES)
            & (pl.col("Type of structure") == TARGET_STRUCTURE)
            & (pl.col("Type of unit") == TARGET_UNIT)
            & (pl.col("REF_DATE").cast(pl.Int64) >= FIRST_SHARED_YEAR)
        )
        .select(
            pl.col("REF_DATE").cast(pl.Int64).alias("year"),
            pl.col("GEO").alias("city"),
            pl.col("DGUID").alias("dguid"),
            pl.col("VALUE").cast(pl.Float64, strict=False).alias("average_rent"),
            pl.col("STATUS").alias("rent_status"),
        )
    )
    vacancy = (
        vacancy_source.filter(
            pl.col("GEO").is_in(TARGET_CITIES)
            & (pl.col("REF_DATE").cast(pl.Int64) >= FIRST_SHARED_YEAR)
        )
        .select(
            pl.col("REF_DATE").cast(pl.Int64).alias("year"),
            pl.col("GEO").alias("city"),
            pl.col("VALUE").cast(pl.Float64, strict=False).alias("vacancy_rate"),
            pl.col("STATUS").alias("vacancy_status"),
        )
    )

    _require_unique_city_year(rent, "Rent data")
    _require_unique_city_year(vacancy, "Vacancy data")

    rent_keys = set(rent.select("year", "city").iter_rows())
    vacancy_keys = set(vacancy.select("year", "city").iter_rows())
    if rent_keys != vacancy_keys:
        missing_rent = sorted(vacancy_keys - rent_keys)
        missing_vacancy = sorted(rent_keys - vacancy_keys)
        raise ValueError(
            "Rent and vacancy city-year keys do not match. "
            f"Missing from rent: {missing_rent}; "
            f"missing from vacancy: {missing_vacancy}"
        )

    return rent.join(vacancy, on=["year", "city"], how="inner").sort(
        "year", "city"
    )
