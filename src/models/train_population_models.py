"""Compare population predictors using the same fixed years for every model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from models import comparison

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DATA_PATH = PROJECT_ROOT / "data/model/edmonton_2br_population_model.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/model/results/population"
TARGET = "average_rent"
FEATURE_SETS = {
    "baseline": ["rent_lag1"],
    "existing_regression": ["rent_lag1", "vacancy_lag1"],
    "population_regression": ["rent_lag1", "vacancy_lag1", "population_growth_lag1"],
}
TRAIN_END_YEAR = 2018
VALIDATION_END_YEAR = 2021
TEST_START_YEAR = 2022
LIMITATIONS = [
    "Only sixteen initial training observations; four test observations.",
    "The 2022–2025 test period has already been inspected in previous work.",
    "Revised exports do not recreate information available at historical forecast dates.",
    "This matching-period comparison uses less training history than the original model.",
    "Coefficients describe associations, not causal effects.",
]


def validate_model_data(data: pd.DataFrame) -> None:
    """Validate the original population experiment's shared years and inputs."""
    comparison.validate_data(data, FEATURE_SETS, list(range(2003, 2026)))


def select_model(validation_mae: dict[str, float]) -> str:
    """Keep the existing validation-only selection interface."""
    return comparison.select_model(validation_mae, FEATURE_SETS)


def train_model(data_path: Path = MODEL_DATA_PATH) -> dict[str, object]:
    """Compare population candidates on identical rows."""
    return comparison.compare_models(
        pd.read_csv(data_path),
        feature_sets=FEATURE_SETS,
        expected_years=list(range(2003, 2026)),
        limitations=LIMITATIONS,
    )


def save_results(results: dict[str, object], output_dir: Path = OUTPUT_DIR) -> None:
    """Save the population experiment separately from production results."""
    comparison.save_results(results, output_dir)


def main() -> None:
    """Print the longer-history comparison without replacing the forecast."""
    results = train_model()
    save_results(results)
    print("Train 2003–2018; validate 2019–2021; test 2022–2025")
    print(
        pd.DataFrame(results["comparison"]).to_string(
            index=False, float_format=lambda x: f"{x:.2f}"
        )
    )
    print(f"Selected using validation only: {results['selected_model']}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
