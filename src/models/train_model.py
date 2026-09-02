"""Train and evaluate the Edmonton two-bedroom rent model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DATA_PATH = PROJECT_ROOT / "data" / "model" / "edmonton_2br_model.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "model" / "results"

FEATURES = ["rent_lag1", "vacancy_lag1"]
TARGET = "average_rent"
REQUIRED_COLUMNS = ["year", TARGET, "vacancy_rate", *FEATURES]

TRAIN_END_YEAR = 2018
VALIDATION_START_YEAR = 2019
VALIDATION_END_YEAR = 2021
TEST_START_YEAR = 2022


def validate_model_data(data: pd.DataFrame) -> None:
    """Validate the input contract expected by the training workflow."""
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    non_numeric_columns = [
        column
        for column in REQUIRED_COLUMNS
        if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if non_numeric_columns:
        raise ValueError(f"Model columns must be numeric: {non_numeric_columns}")

    if data[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Missing values found in required model columns")

    if not np.isfinite(data[REQUIRED_COLUMNS].to_numpy()).all():
        raise ValueError("Non-finite values found in required model columns")

    if not data["year"].is_unique:
        raise ValueError("Duplicate years found in model dataset")

    if not data["year"].is_monotonic_increasing:
        raise ValueError("Model data must be ordered chronologically")

    splits = {
        "Training": data[data["year"] <= TRAIN_END_YEAR],
        "Validation": data[
            data["year"].between(VALIDATION_START_YEAR, VALIDATION_END_YEAR)
        ],
        "Test": data[data["year"] >= TEST_START_YEAR],
    }
    for split_name, split in splits.items():
        if split.empty:
            raise ValueError(f"{split_name} split is empty")


def train_model(data_path: Path = MODEL_DATA_PATH) -> dict[str, object]:
    """Train candidate models and return evaluation and forecast results."""
    data = pd.read_csv(data_path)
    validate_model_data(data)

    train = data[data["year"] <= TRAIN_END_YEAR]
    validation = data[data["year"].between(VALIDATION_START_YEAR, VALIDATION_END_YEAR)]
    test = data[data["year"] >= TEST_START_YEAR]

    baseline_validation_predictions = validation["rent_lag1"]
    baseline_validation_mae = mean_absolute_error(
        validation[TARGET], baseline_validation_predictions
    )

    regression_model = LinearRegression().fit(train[FEATURES], train[TARGET])
    regression_validation_predictions = regression_model.predict(validation[FEATURES])
    regression_validation_mae = mean_absolute_error(
        validation[TARGET], regression_validation_predictions
    )

    selected_model = (
        "linear_regression"
        if regression_validation_mae < baseline_validation_mae
        else "baseline"
    )

    train_validation = data[data["year"] <= VALIDATION_END_YEAR]
    if selected_model == "linear_regression":
        evaluation_model = LinearRegression().fit(
            train_validation[FEATURES], train_validation[TARGET]
        )
        test_predictions = evaluation_model.predict(test[FEATURES])
    else:
        test_predictions = test["rent_lag1"].to_numpy()

    test_mae = mean_absolute_error(test[TARGET], test_predictions)
    test_rmse = np.sqrt(mean_squared_error(test[TARGET], test_predictions))
    test_r2 = r2_score(test[TARGET], test_predictions)

    production_model = None
    latest_row = data.iloc[-1]
    if selected_model == "linear_regression":
        production_model = LinearRegression().fit(data[FEATURES], data[TARGET])
        future_features = pd.DataFrame(
            [
                {
                    "rent_lag1": latest_row[TARGET],
                    "vacancy_lag1": latest_row["vacancy_rate"],
                }
            ]
        )
        forecast_rent = production_model.predict(future_features)[0]
    else:
        forecast_rent = latest_row[TARGET]

    prediction_rows = []
    for year, actual_rent, predicted_rent in zip(
        test["year"], test[TARGET], test_predictions
    ):
        prediction_rows.append(
            {
                "year": int(year),
                "average_rent": float(actual_rent),
                "predicted_rent": round(float(predicted_rent), 2),
                "absolute_error": round(float(abs(actual_rent - predicted_rent)), 2),
            }
        )

    results: dict[str, object] = {
        "selected_model": selected_model,
        "baseline_validation_mae": float(baseline_validation_mae),
        "regression_validation_mae": float(regression_validation_mae),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),
        "test_years": test["year"].astype(int).tolist(),
        "test_predictions": prediction_rows,
        "forecast_year": int(latest_row["year"]) + 1,
        "forecast_rent": float(forecast_rent),
        "coefficients": None,
        "intercept": None,
    }

    if production_model is not None:
        results["coefficients"] = {
            feature: round(float(coefficient), 4)
            for feature, coefficient in zip(FEATURES, production_model.coef_)
        }
        results["intercept"] = round(float(production_model.intercept_), 4)

    return results


def save_results(results: dict[str, object], output_dir: Path = OUTPUT_DIR) -> None:
    """Save the model summary and test predictions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame(results["test_predictions"])
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    summary = {
        "selected_model": results["selected_model"],
        "test_mae": round(float(results["test_mae"]), 2),
        "test_rmse": round(float(results["test_rmse"]), 2),
        "test_r2": round(float(results["test_r2"]), 3),
        "coefficients": results["coefficients"],
        "intercept": results["intercept"],
        "validation": {
            "baseline_mae": round(float(results["baseline_validation_mae"]), 2),
            "linear_regression_mae": round(
                float(results["regression_validation_mae"]), 2
            ),
        },
        "test_years": results["test_years"],
        "forecast": {
            "year": results["forecast_year"],
            "predicted_rent": round(float(results["forecast_rent"]), 2),
        },
    }
    (output_dir / "model_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Run training, save artifacts, and print a concise summary."""
    results = train_model()
    save_results(results)

    print(f"Selected model: {results['selected_model']}")
    print(
        "Test metrics: "
        f"MAE={results['test_mae']:.2f}, "
        f"RMSE={results['test_rmse']:.2f}, "
        f"R²={results['test_r2']:.3f}"
    )
    print(f"{results['forecast_year']} forecast: ${results['forecast_rent']:,.2f}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
