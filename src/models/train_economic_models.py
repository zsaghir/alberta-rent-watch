"""Compare economic predictors using the same fixed years for every model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    """Require complete, ordered observations for the shared comparison period."""
    required = ["year", TARGET, *FEATURE_SETS["full_regression"]]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if any(not pd.api.types.is_numeric_dtype(data[column]) for column in required):
        raise ValueError("Required model columns must be numeric")
    if not np.isfinite(data[required].to_numpy()).all():
        raise ValueError("Required model columns contain missing or non-finite values")
    if data["year"].tolist() != list(range(2012, 2026)):
        raise ValueError(
            "Model years must be unique, ordered and consecutive: 2012–2025"
        )


def select_model(validation_mae: dict[str, float]) -> str:
    """Prefer lower validation error, then fewer inputs, then declared order."""
    order = list(FEATURE_SETS)
    return min(
        order,
        key=lambda name: (
            validation_mae[name],
            len(FEATURE_SETS[name]),
            order.index(name),
        ),
    )


def train_model(data_path: Path = MODEL_DATA_PATH) -> dict[str, object]:
    """Compare all candidates without selecting a winner from test results."""
    data = pd.read_csv(data_path)
    validate_model_data(data)
    train = data[data["year"] <= TRAIN_END_YEAR]
    validation = data[data["year"].between(TRAIN_END_YEAR + 1, VALIDATION_END_YEAR)]
    train_validation = data[data["year"] <= VALIDATION_END_YEAR]
    test = data[data["year"] >= TEST_START_YEAR]

    validation_mae = {}
    for name, features in FEATURE_SETS.items():
        predictions = validation["rent_lag1"].to_numpy()
        if name != "baseline":
            model = LinearRegression().fit(train[features], train[TARGET])
            predictions = model.predict(validation[features])
        validation_mae[name] = float(
            mean_absolute_error(validation[TARGET], predictions)
        )

    # Lock selection before looking at test errors.
    selected = select_model(validation_mae)
    comparison = []
    prediction_rows = []
    parameters = {}
    for name, features in FEATURE_SETS.items():
        predictions = test["rent_lag1"].to_numpy()
        parameters[name] = {"coefficients": None, "intercept": None}
        if name != "baseline":
            # Fit once through 2021; do not retrain on any test-year targets.
            model = LinearRegression().fit(
                train_validation[features], train_validation[TARGET]
            )
            predictions = model.predict(test[features])
            parameters[name] = {
                "coefficients": dict(zip(features, map(float, model.coef_))),
                "intercept": float(model.intercept_),
            }
        comparison.append(
            {
                "model": name,
                "validation_mae": validation_mae[name],
                "test_mae": float(mean_absolute_error(test[TARGET], predictions)),
                "test_rmse": float(
                    np.sqrt(mean_squared_error(test[TARGET], predictions))
                ),
                "test_r2": float(r2_score(test[TARGET], predictions)),
                "selected": name == selected,
            }
        )
        for year, actual, predicted in zip(test["year"], test[TARGET], predictions):
            prediction_rows.append(
                {
                    "model": name,
                    "year": int(year),
                    "average_rent": float(actual),
                    "predicted_rent": float(predicted),
                    "absolute_error": float(abs(actual - predicted)),
                }
            )

    reference = next(
        row["test_mae"] for row in comparison if row["model"] == "existing_regression"
    )
    for row in comparison:
        row["mae_improvement_pct"] = (
            (reference - row["test_mae"]) / reference * 100 if reference != 0 else None
        )

    return {
        "selected_model": selected,
        "feature_sets": FEATURE_SETS,
        "splits": {
            name: {"years": frame["year"].astype(int).tolist(), "row_count": len(frame)}
            for name, frame in (
                ("train", train),
                ("validation", validation),
                ("evaluation_fit", train_validation),
                ("test", test),
            )
        },
        "evaluation_parameters": parameters,
        "comparison": comparison,
        "test_predictions": prediction_rows,
        "limitations": LIMITATIONS,
    }


def save_results(results: dict[str, object], output_dir: Path = OUTPUT_DIR) -> None:
    """Save full-precision metrics and predictions in a separate experiment folder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results["comparison"]).to_csv(
        output_dir / "model_comparison.csv", index=False
    )
    pd.DataFrame(results["test_predictions"]).to_csv(
        output_dir / "test_predictions.csv", index=False
    )
    summary = {
        key: value for key, value in results.items() if key != "test_predictions"
    }
    (output_dir / "model_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


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
