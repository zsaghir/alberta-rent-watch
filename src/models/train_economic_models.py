"""Compare economic predictors using the same fixed years for every model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from models import comparison

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DATA_PATH = PROJECT_ROOT / "data/model/edmonton_2br_economic_model.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/model/results/economic"
TARGET = "average_rent"
FEATURE_SETS = {
    "baseline": ["rent_lag1"],
    "existing_regression": ["rent_lag1", "vacancy_lag1"],
    "population_regression": ["rent_lag1", "vacancy_lag1", "population_growth_lag1"],
    "unemployment_regression": ["rent_lag1", "vacancy_lag1", "unemployment_lag1"],
    "full_regression": [
        "rent_lag1",
        "vacancy_lag1",
        "population_growth_lag1",
        "unemployment_lag1",
    ],
}
TRAIN_END_YEAR = 2018
VALIDATION_END_YEAR = 2021
TEST_START_YEAR = 2022
LIMITATIONS = [
    "Only seven initial training observations; four test observations.",
    "The 2022–2025 test period has already been inspected in previous work.",
    "Revised exports do not recreate information available at historical forecast dates.",
    "This matching-period comparison uses less training history than the original model.",
    "Coefficients describe associations, not causal effects.",
]


def validate_model_data(data: pd.DataFrame) -> None:
    """Validate the original economic experiment's shared years and inputs."""
    comparison.validate_data(data, FEATURE_SETS, list(range(2012, 2026)))


def select_model(validation_mae: dict[str, float]) -> str:
    """Keep the existing validation-only selection interface."""
    return comparison.select_model(validation_mae, FEATURE_SETS)


def train_model(data_path: Path = MODEL_DATA_PATH) -> dict[str, object]:
    """Compare economic candidates on identical rows."""
    return comparison.compare_models(
        pd.read_csv(data_path),
        feature_sets=FEATURE_SETS,
        expected_years=list(range(2012, 2026)),
        limitations=LIMITATIONS,
    )


def save_results(results: dict[str, object], output_dir: Path = OUTPUT_DIR) -> None:
    """Save the economic experiment separately from production results."""
    comparison.save_results(results, output_dir)


def main() -> None:
    """Run the experiment and show errors in dollars per month."""
    results = train_model()
    save_results(results)
    print(
        "Matching-period comparison: train 2012–2018; validate 2019–2021; test 2022–2025"
    )
    print(
        pd.DataFrame(results["comparison"]).to_string(
            index=False, float_format=lambda x: f"{x:.2f}"
        )
    )
    print(f"Selected using validation only: {results['selected_model']}")
    for row in results["comparison"][2:]:
        improved = (
            row["mae_improvement_pct"] is not None and row["mae_improvement_pct"] > 0
        )
        print(
            f"{row['model']}: {'reduced' if improved else 'did not reduce'} test MAE versus existing regression."
        )
    print("Exploratory results: only seven initial training rows and four test years.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
