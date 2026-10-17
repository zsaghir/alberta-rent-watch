import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.build_model_data import (
    ECONOMIC_COLUMNS,
    MASTER_PATH,
    add_economic_features,
    build_economic_model_data,
    build_model_data,
)
from models.train_economic_models import (
    FEATURE_SETS,
    MODEL_DATA_PATH,
    save_results,
    select_model,
    train_model,
    validate_model_data,
)

ROOT = Path(__file__).parents[1]


def test_growth_and_lags_use_previous_year():
    master = pd.read_csv(MASTER_PATH)
    master["population"] = [np.nan] + [1000 * 1.1**i for i in range(25)]
    master.loc[master.year == 2024, "unemployment_rate"] = 9.0
    result = add_economic_features(master)
    assert result.columns.tolist() == ECONOMIC_COLUMNS
    assert result.year.tolist() == list(range(2012, 2026))
    row = result.iloc[-1]
    assert row.population_growth_lag1 == pytest.approx(10)
    assert row.unemployment_lag1 == 9
    assert row.rent_lag1 == 1529
    assert row.vacancy_lag1 == 3
    # Changing this year's population cannot change this year's lagged input.
    master.loc[master.year == 2025, "population"] *= 2
    assert add_economic_features(master).iloc[
        -1
    ].population_growth_lag1 == pytest.approx(10)


@pytest.mark.parametrize(
    "problem", ["missing_column", "gap", "duplicate", "nan", "inf"]
)
def test_feature_builder_rejects_bad_inputs(problem):
    master = pd.read_csv(MASTER_PATH)
    if problem == "missing_column":
        master = master.drop(columns="population")
    elif problem == "gap":
        master = master[master.year != 2010]
    elif problem == "duplicate":
        master.loc[1, "year"] = 2000
    else:
        master.loc[master.year == 2011, "unemployment_rate"] = float(problem)
    with pytest.raises(ValueError):
        add_economic_features(master)


@pytest.mark.parametrize(
    "problem", ["missing_column", "gap", "duplicate", "nan", "inf", "unordered"]
)
def test_training_rejects_bad_inputs(problem):
    data = pd.read_csv(MODEL_DATA_PATH)
    if problem == "missing_column":
        data = data.drop(columns="population_growth_lag1")
    elif problem == "gap":
        data = data.iloc[1:]
    elif problem == "duplicate":
        data.loc[1, "year"] = 2012
    elif problem == "unordered":
        data = data.iloc[::-1]
    else:
        data.loc[0, "rent_lag1"] = float(problem)
    with pytest.raises(ValueError):
        validate_model_data(data)


def test_shared_splits_baseline_and_metrics():
    result = train_model()
    assert result["splits"]["train"]["years"] == list(range(2012, 2019))
    assert result["splits"]["validation"]["years"] == [2019, 2020, 2021]
    assert result["splits"]["evaluation_fit"]["years"] == list(range(2012, 2022))
    assert result["splits"]["test"]["years"] == [2022, 2023, 2024, 2025]
    predictions = pd.DataFrame(result["test_predictions"])
    for _, frame in predictions.groupby("model"):
        assert frame.year.tolist() == [2022, 2023, 2024, 2025]
    data = pd.read_csv(MODEL_DATA_PATH)
    assert (
        predictions[predictions.model == "baseline"].predicted_rent.tolist()
        == data.loc[data.year >= 2022, "rent_lag1"].tolist()
    )
    for row in result["comparison"]:
        frame = predictions[predictions.model == row["model"]]
        assert row["test_mae"] == pytest.approx(frame.absolute_error.mean())
    assert result["evaluation_parameters"]["baseline"] == {
        "coefficients": None,
        "intercept": None,
    }


def test_selection_ignores_test_winner(tmp_path):
    rng = np.random.default_rng(12)
    data = pd.DataFrame({"year": range(2012, 2026)})
    for name in FEATURE_SETS["full_regression"]:
        data[name] = rng.uniform(1, 10, len(data))
    data["average_rent"] = 10 * data["population_growth_lag1"] + 2 * data["rent_lag1"]
    validation = data.year.between(2019, 2021)
    data.loc[validation, "average_rent"] = data.loc[validation, "rent_lag1"]
    path = tmp_path / "synthetic.csv"
    data.to_csv(path, index=False)
    result = train_model(path)
    assert result["selected_model"] == "baseline"
    best_test = min(result["comparison"], key=lambda row: row["test_mae"])["model"]
    assert best_test != "baseline"
    # Changing test targets must not change validation selection or fitted predictions.
    data.loc[data.year >= 2022, "average_rent"] += 10000
    data.to_csv(path, index=False)
    changed = train_model(path)
    assert changed["selected_model"] == result["selected_model"]
    assert changed["evaluation_parameters"] == result["evaluation_parameters"]
    assert [r["predicted_rent"] for r in changed["test_predictions"]] == [
        r["predicted_rent"] for r in result["test_predictions"]
    ]


def test_tie_break_and_zero_reference(tmp_path):
    assert select_model(dict.fromkeys(FEATURE_SETS, 0.0)) == "baseline"
    scores = dict.fromkeys(FEATURE_SETS, 10.0)
    scores.update(population_regression=1.0, unemployment_regression=1.0)
    assert select_model(scores) == "population_regression"
    data = pd.read_csv(MODEL_DATA_PATH)
    data["average_rent"] = 0.0
    data.to_csv(tmp_path / "zero.csv", index=False)
    result = train_model(tmp_path / "zero.csv")
    assert all(row["mae_improvement_pct"] is None for row in result["comparison"])
    save_results(result, tmp_path / "results")
    json.loads((tmp_path / "results/model_summary.json").read_text())


def assert_json_matches(actual, expected):
    """Keep structure exact, allowing only tiny floating-point differences."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_json_matches(actual[key], expected[key])
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert_json_matches(actual_item, expected_item)
    elif isinstance(expected, float):
        assert isinstance(actual, float)
        assert actual == pytest.approx(expected, rel=1e-10, abs=1e-9)
    else:
        assert type(actual) is type(expected)
        assert actual == expected


def assert_artifact_matches(actual_path, expected_path):
    # Mac and Linux linear algebra can differ in the last decimal places.
    if actual_path.suffix == ".json":
        assert_json_matches(
            json.loads(actual_path.read_text()),
            json.loads(expected_path.read_text()),
        )
    else:
        actual = pd.read_csv(actual_path)
        expected = pd.read_csv(expected_path)
        assert actual.columns.tolist() == expected.columns.tolist()
        for column in expected:
            pd.testing.assert_series_equal(
                actual[column],
                expected[column],
                check_exact=not pd.api.types.is_float_dtype(expected[column]),
                rtol=1e-10,
                atol=1e-9,
            )


def test_artifact_comparison_accepts_roundoff_but_rejects_changed_results(tmp_path):
    expected = {
        "selected": True,
        "year": 2025,
        "model": "baseline",
        "mae": 89.91060441047932,
    }
    assert_json_matches({**expected, "mae": 89.9106044104796}, expected)
    for change in (
        {"mae": 90.0},
        {"selected": False},
        {"year": 2024},
        {"model": "full_regression"},
    ):
        with pytest.raises(AssertionError):
            assert_json_matches({**expected, **change}, expected)

    expected_path = tmp_path / "expected.csv"
    actual_path = tmp_path / "actual.csv"
    frame = pd.DataFrame([expected])
    frame.to_csv(expected_path, index=False)
    frame.assign(mae=89.9106044104796).to_csv(actual_path, index=False)
    assert_artifact_matches(actual_path, expected_path)
    for change in (
        {"mae": 90.0},
        {"selected": False},
        {"year": 2024},
        {"model": "full_regression"},
    ):
        frame.assign(**change).to_csv(actual_path, index=False)
        with pytest.raises(AssertionError):
            assert_artifact_matches(actual_path, expected_path)


def test_reproducibility_and_existing_artifacts(tmp_path):
    build_model_data(output_path=tmp_path / "original.csv")
    assert (tmp_path / "original.csv").read_bytes() == (
        ROOT / "data/model/edmonton_2br_model.csv"
    ).read_bytes()
    build_economic_model_data(output_path=tmp_path / "economic.csv")
    assert_artifact_matches(tmp_path / "economic.csv", MODEL_DATA_PATH)
    result = train_model()
    save_results(result, tmp_path / "one")
    save_results(result, tmp_path / "two")
    for filename in (
        "model_comparison.csv",
        "test_predictions.csv",
        "model_summary.json",
    ):
        assert (tmp_path / "one" / filename).read_bytes() == (
            tmp_path / "two" / filename
        ).read_bytes()
        assert_artifact_matches(
            tmp_path / "one" / filename,
            ROOT / "data/model/results/economic" / filename,
        )
    from models.train_model import save_results as save_original
    from models.train_model import train_model as train_original

    save_original(train_original(), tmp_path / "original")
    for filename in ("model_summary.json", "test_predictions.csv"):
        assert_artifact_matches(
            tmp_path / "original" / filename,
            ROOT / "data/model/results" / filename,
        )


def test_import_has_no_side_effects(tmp_path):
    result_dir = ROOT / "data/model/results"
    before = {p: p.stat().st_mtime_ns for p in result_dir.rglob("*") if p.is_file()}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import models.train_economic_models; import data.build_model_data",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ""
    assert list(tmp_path.iterdir()) == []
    assert before == {
        p: p.stat().st_mtime_ns for p in result_dir.rglob("*") if p.is_file()
    }
