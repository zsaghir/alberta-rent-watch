"""Build the canonical Edmonton rent dataset from the raw CMHC files."""

from pathlib import Path

import polars as pl

from alberta_rent_watch.cmhc import load_cmhc_data

PROJECT_ROOT = Path(__file__).parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "edmonton_2br_master.csv"

FIRST_YEAR = 2000
LAST_YEAR = 2025
EXPECTED_ROW_COUNT = LAST_YEAR - FIRST_YEAR + 1


def validate_master(master: pl.DataFrame) -> None:
    """Check that the master dataset is complete and internally consistent."""
    expected_columns = ["year", "average_rent", "vacancy_rate"]
    if master.columns != expected_columns:
        raise ValueError(f"Unexpected master columns: {master.columns}")

    if master.height != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ROW_COUNT} rows, but found {master.height}"
        )

    if master["year"].is_duplicated().any():
        raise ValueError("The master dataset contains duplicate years")

    expected_years = list(range(FIRST_YEAR, LAST_YEAR + 1))
    actual_years = master["year"].to_list()
    if actual_years != expected_years:
        raise ValueError(
            f"Expected years {FIRST_YEAR}–{LAST_YEAR}, but found {actual_years}"
        )

    if master["average_rent"].null_count() > 0:
        raise ValueError("The master dataset contains missing rent values")

    if master["vacancy_rate"].null_count() > 0:
        raise ValueError("The master dataset contains missing vacancy values")

    if (master["average_rent"] <= 0).any():
        raise ValueError("Average rent must be greater than zero")

    invalid_vacancy = (master["vacancy_rate"] < 0) | (master["vacancy_rate"] > 100)
    if invalid_vacancy.any():
        raise ValueError("Vacancy rate must be between 0 and 100")


def build_master(
    raw_dir: Path = RAW_DIR,
    output_path: Path = OUTPUT_PATH,
) -> pl.DataFrame:
    """Clean, validate, and save the canonical master dataset."""
    cleaned_data = load_cmhc_data(raw_dir)

    master = (
        cleaned_data.select("year", "average_rent", "vacancy_rate")
        .with_columns(
            pl.col("year").cast(pl.Int64),
            pl.col("average_rent").cast(pl.Float64),
            pl.col("vacancy_rate").cast(pl.Float64),
        )
        .sort("year")
    )

    validate_master(master)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.write_csv(output_path)
    return master


if __name__ == "__main__":
    result = build_master()
    print(f"Created {OUTPUT_PATH} with {result.height} annual observations.")
