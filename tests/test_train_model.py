import json
import os
import subprocess
import sys
from math import isfinite
from pathlib import Path

import pandas as pd
import pytest

from models.train_model import save_results, train_model, validate_model_data

PROJECT_ROOT = Path(__file__).parents[1]
MODEL_DATA_PATH = PROJECT_ROOT / "data" / "model" / "edmonton_2br_model.csv"
RESULTS_DIR = PROJECT_ROOT / "data" / "model" / "results"

METRIC_KEYS = [
    "baseline_validation_mae",
    "regression_validation_mae",
    "test_mae",
    "test_rmse",
    "test_r2",
]


def make_model_data(model: str = "linear_regression") -> pd.DataFrame:
    years = list(range(2001, 2026))
    rent_lag = [500.0 + 10 * index for index in range(len(years))]
    vacancy_lag = [2.0 + 0.1 * (index % 4) for index in range(len(years))]
    if model == "linear_regression":
        average_rent = [
            2 * rent + 3 * vacancy for rent, vacancy in zip(rent_lag, vacancy_lag)
        ]
    else:
        average_rent = rent_lag

    return pd.DataFrame(
        {
            "year": years,
            "average_rent": average_rent,
            "vacancy_rate": vacancy_lag,
            "rent_lag1": rent_lag,
            "vacancy_lag1": vacancy_lag,
        }
    )


def write_model_data(tmp_path: Path, data: pd.DataFrame) -> Path:
    path = tmp_path / "model.csv"
    data.to_csv(path, index=False)
    return path


def test_import_has_no_output_or_file_writes() -> None:
    existing_files = {
        path: path.stat().st_mtime_ns
        for path in RESULTS_DIR.glob("*")
        if path.is_file()
    }
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    result = subprocess.run(
        [sys.executable, "-c", "import models.train_model"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert {
        path: path.stat().st_mtime_ns
        for path in RESULTS_DIR.glob("*")
        if path.is_file()
    } == existing_files


def test_train_model_returns_complete_finite_results() -> None:
    results = train_model(MODEL_DATA_PATH)

    assert results["selected_model"] in {"baseline", "linear_regression"}
    assert all(isfinite(results[key]) for key in METRIC_KEYS)
    assert results["test_years"] == [2022, 2023, 2024, 2025]
    assert len(results["test_predictions"]) == 4
    assert results["forecast_year"] == 2026
    assert isfinite(results["forecast_rent"])
    assert set(results["coefficients"]) == {"rent_lag1", "vacancy_lag1"}
    assert isfinite(results["intercept"])


@pytest.mark.parametrize(
    ("model", "expected"),
    [("linear_regression", "linear_regression"), ("baseline", "baseline")],
)
def test_selects_model_with_lower_validation_error(
    tmp_path: Path, model: str, expected: str
) -> None:
    path = write_model_data(tmp_path, make_model_data(model))

    results = train_model(path)

    assert results["selected_model"] == expected
    if expected == "linear_regression":
        assert set(results["coefficients"]) == {"rent_lag1", "vacancy_lag1"}
        assert isfinite(results["intercept"])
    else:
        assert results["coefficients"] is None
        assert results["intercept"] is None


def test_rejects_missing_columns() -> None:
    data = make_model_data().drop(columns="rent_lag1")

    with pytest.raises(ValueError, match="Missing required columns.*rent_lag1"):
        validate_model_data(data)


@pytest.mark.parametrize(
    "column", ["average_rent", "vacancy_rate", "rent_lag1", "vacancy_lag1"]
)
def test_rejects_missing_values(column: str) -> None:
    data = make_model_data()
    data.loc[0, column] = None

    with pytest.raises(ValueError, match="Missing values"):
        validate_model_data(data)


def test_rejects_duplicate_years() -> None:
    data = make_model_data()
    data.loc[1, "year"] = data.loc[0, "year"]

    with pytest.raises(ValueError, match="Duplicate years"):
        validate_model_data(data)


def test_rejects_unordered_years() -> None:
    data = make_model_data()
    data.loc[[0, 1], "year"] = data.loc[[1, 0], "year"].to_numpy()

    with pytest.raises(ValueError, match="ordered chronologically"):
        validate_model_data(data)


@pytest.mark.parametrize(
    ("years", "missing_split"),
    [
        (range(2019, 2026), "Training"),
        (range(2001, 2019), "Validation"),
        (range(2001, 2022), "Test"),
    ],
)
def test_rejects_empty_splits(years: range, missing_split: str) -> None:
    data = make_model_data()
    data = data[data["year"].isin(years)]

    with pytest.raises(ValueError, match=f"{missing_split} split is empty"):
        validate_model_data(data)


def test_save_results_writes_expected_json_and_csv(tmp_path: Path) -> None:
    results = train_model(MODEL_DATA_PATH)

    save_results(results, tmp_path)

    summary = json.loads((tmp_path / "model_summary.json").read_text())
    predictions = pd.read_csv(tmp_path / "test_predictions.csv")
    assert summary["selected_model"] == results["selected_model"]
    assert all(isfinite(summary[key]) for key in ["test_mae", "test_rmse", "test_r2"])
    assert set(summary["coefficients"]) == {"rent_lag1", "vacancy_lag1"}
    assert isfinite(summary["intercept"])
    assert summary["forecast"]["year"] == 2026
    assert summary["test_years"] == [2022, 2023, 2024, 2025]
    assert predictions.columns.tolist() == [
        "year",
        "average_rent",
        "predicted_rent",
        "absolute_error",
    ]
    assert predictions["year"].tolist() == [2022, 2023, 2024, 2025]


def test_save_results_uses_null_parameters_for_baseline(tmp_path: Path) -> None:
    data_path = write_model_data(tmp_path, make_model_data("baseline"))
    results = train_model(data_path)
    output_dir = tmp_path / "results"

    save_results(results, output_dir)

    summary = json.loads((output_dir / "model_summary.json").read_text())
    assert summary["selected_model"] == "baseline"
    assert summary["coefficients"] is None
    assert summary["intercept"] is None
