"""Build the model-ready Edmonton rent dataset with one-year lag features."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "edmonton_2br_master.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "model" / "edmonton_2br_model.csv"

MASTER_COLUMNS = ["year", "average_rent", "vacancy_rate"]
LAG_COLUMNS = ["rent_lag1", "vacancy_lag1"]
MODEL_COLUMNS = [*MASTER_COLUMNS, *LAG_COLUMNS]


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


if __name__ == "__main__":
    result = build_model_data()
    print(f"Created {OUTPUT_PATH} with {len(result)} model-ready rows.")
