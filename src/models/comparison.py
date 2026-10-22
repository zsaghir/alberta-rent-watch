"""Shared fixed-split comparison for the economic and population experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "average_rent"
TRAIN_END_YEAR = 2018
VALIDATION_END_YEAR = 2021
TEST_START_YEAR = 2022


def validate_data(
    data: pd.DataFrame, feature_sets: dict[str, list[str]], expected_years: list[int]
) -> None:
    """Require complete, ordered observations for the shared comparison period."""
    required = [
        "year",
        TARGET,
        *dict.fromkeys(
            feature for features in feature_sets.values() for feature in features
        ),
    ]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if any(not pd.api.types.is_numeric_dtype(data[column]) for column in required):
        raise ValueError("Required model columns must be numeric")
    if not np.isfinite(data[required].to_numpy()).all():
        raise ValueError("Required model columns contain missing or non-finite values")
    if data["year"].tolist() != expected_years:
        raise ValueError(
            f"Model years must be unique, ordered and consecutive: {expected_years}"
        )


def select_model(
    validation_mae: dict[str, float], feature_sets: dict[str, list[str]]
) -> str:
    """Prefer lower validation error, then fewer inputs, then declared order."""
    order = list(feature_sets)
    return min(
        order,
        key=lambda name: (
            validation_mae[name],
            len(feature_sets[name]),
            order.index(name),
        ),
    )


def compare_models(
    data: pd.DataFrame,
    *,
    feature_sets: dict[str, list[str]],
    expected_years: list[int],
    limitations: list[str],
) -> dict[str, object]:
    """Compare all candidates without selecting a winner from test results."""
    validate_data(data, feature_sets, expected_years)
    train = data[data["year"] <= TRAIN_END_YEAR]
    validation = data[data["year"].between(TRAIN_END_YEAR + 1, VALIDATION_END_YEAR)]
    train_validation = data[data["year"] <= VALIDATION_END_YEAR]
    test = data[data["year"] >= TEST_START_YEAR]

    validation_mae = {}
    for name, features in feature_sets.items():
        predictions = validation["rent_lag1"].to_numpy()
        if name != "baseline":
            model = LinearRegression().fit(train[features], train[TARGET])
            predictions = model.predict(validation[features])
        validation_mae[name] = float(
            mean_absolute_error(validation[TARGET], predictions)
        )

    # Lock selection before looking at test errors.
    selected = select_model(validation_mae, feature_sets)
    comparison = []
    prediction_rows = []
    parameters = {}
    for name, features in feature_sets.items():
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
        "feature_sets": feature_sets,
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
        "limitations": limitations,
    }


def save_results(results: dict[str, object], output_dir: Path) -> None:
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
