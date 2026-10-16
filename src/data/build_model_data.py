"""Build the model-ready Edmonton rent dataset with one-year lag features."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "edmonton_2br_master.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "model" / "edmonton_2br_model.csv"

MASTER_COLUMNS = ["year", "average_rent", "vacancy_rate"]
LAG_COLUMNS = ["rent_lag1", "vacancy_lag1"]
MODEL_COLUMNS = [*MASTER_COLUMNS, *LAG_COLUMNS]

ECONOMIC_OUTPUT_PATH = OUTPUT_PATH.with_name("edmonton_2br_economic_model.csv")
ECONOMIC_MASTER_COLUMNS = [*MASTER_COLUMNS, "population", "unemployment_rate"]
ECONOMIC_FEATURES = [*LAG_COLUMNS, "population_growth_lag1", "unemployment_lag1"]
ECONOMIC_COLUMNS = [*ECONOMIC_MASTER_COLUMNS, "population_growth", *ECONOMIC_FEATURES]


def add_lag_features(master: pd.DataFrame) -> pd.DataFrame:
    """Add prior-year rent and vacancy and drop the year without a prior year."""
    missing_columns = [
        column for column in MASTER_COLUMNS if column not in master.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    ordered = master[MASTER_COLUMNS].sort_values("year").reset_index(drop=True)
    expected_years = list(range(ordered["year"].iloc[0], ordered["year"].iloc[-1] + 1))
    if ordered["year"].tolist() != expected_years:
        raise ValueError("Master years must be unique and consecutive")

    ordered["rent_lag1"] = ordered["average_rent"].shift(1)
    ordered["vacancy_lag1"] = ordered["vacancy_rate"].shift(1)
    return ordered.dropna(subset=LAG_COLUMNS).reset_index(drop=True)


def build_model_data(
    master_path: Path = MASTER_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Read the master dataset, add lag features, and save the model dataset."""
    model_data = add_lag_features(pd.read_csv(master_path))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_data.to_csv(output_path, index=False)
    return model_data


def add_economic_features(master: pd.DataFrame) -> pd.DataFrame:
    """Create economic predictors without filling gaps or using future years."""
    missing = [column for column in ECONOMIC_MASTER_COLUMNS if column not in master]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    ordered = master[ECONOMIC_MASTER_COLUMNS].sort_values("year").reset_index(drop=True)
    if ordered["year"].tolist() != list(range(2000, 2026)):
        raise ValueError("Master years must be unique and consecutive from 2000–2025")
    for column in ECONOMIC_MASTER_COLUMNS:
        if not pd.api.types.is_numeric_dtype(ordered[column]):
            raise ValueError(f"{column} must be numeric")
    for column, start in (
        ("average_rent", 2000),
        ("vacancy_rate", 2000),
        ("population", 2001),
        ("unemployment_rate", 2011),
    ):
        covered = ordered.loc[ordered["year"] >= start, column]
        if not np.isfinite(covered.to_numpy()).all():
            raise ValueError(f"Missing or non-finite {column} within coverage")
        if ordered.loc[ordered["year"] < start, column].notna().any():
            raise ValueError(f"{column} must be missing before {start}")
    if (ordered["population"].dropna() <= 0).any():
        raise ValueError("Population must be positive")

    # Calculate growth first: growth in 2024 compares populations in 2023 and 2024.
    ordered["population_growth"] = (
        ordered["population"] / ordered["population"].shift(1) - 1
    ) * 100
    # Shift inputs so predicting 2025 uses observations from 2024.
    for source, target in (
        ("average_rent", "rent_lag1"),
        ("vacancy_rate", "vacancy_lag1"),
        ("population_growth", "population_growth_lag1"),
        ("unemployment_rate", "unemployment_lag1"),
    ):
        ordered[target] = ordered[source].shift(1)

    # Unemployment begins in 2011, so its first lagged prediction year is 2012.
    result = ordered.loc[ordered["year"] >= 2012, ECONOMIC_COLUMNS].reset_index(
        drop=True
    )
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Economic dataset contains missing or non-finite values")
    return result


def build_economic_model_data(
    master_path: Path = MASTER_PATH,
    output_path: Path = ECONOMIC_OUTPUT_PATH,
) -> pd.DataFrame:
    """Save a separate economic dataset, preserving the original model data."""
    result = add_economic_features(pd.read_csv(master_path))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    """Build both model datasets from the same master history."""
    result = build_model_data()
    print(f"Created {OUTPUT_PATH} with {len(result)} model-ready rows.")
    economic = build_economic_model_data()
    print(f"Created {ECONOMIC_OUTPUT_PATH} with {len(economic)} economic rows.")


if __name__ == "__main__":
    main()
