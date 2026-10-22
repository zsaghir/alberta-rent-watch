import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.build_model_data import (
    MASTER_PATH,
    add_population_features,
    build_population_model_data,
)
from models.train_population_models import (
    MODEL_DATA_PATH,
    OUTPUT_DIR,
    save_results,
    train_model,
    validate_model_data,
)

ROOT = Path(__file__).parents[1]


def test_population_features_ignore_unemployment_and_align_years():
    master = pd.read_csv(MASTER_PATH).drop(columns="unemployment_rate")
    result = add_population_features(master)
    assert result.year.tolist() == list(range(2003, 2026))
    first = result.iloc[0]
    assert first.population_growth_lag1 == pytest.approx((985447 / 963062 - 1) * 100)
    assert first.rent_lag1 == 710
    assert first.vacancy_lag1 == 1.6
    master.loc[master.year == 2003, "population"] *= 2
    assert (
        add_population_features(master).iloc[0].population_growth_lag1
        == first.population_growth_lag1
    )
    assert "unemployment_rate" not in result


@pytest.mark.parametrize(
    "problem", ["missing_column", "gap", "duplicate", "nan", "inf", "zero"]
)
def test_population_builder_rejects_invalid_data(problem):
    data = pd.read_csv(MASTER_PATH)
    if problem == "missing_column":
        data = data.drop(columns="population")
    elif problem == "gap":
        data = data[data.year != 2002]
    elif problem == "duplicate":
        data.loc[1, "year"] = 2000
    else:
        data.loc[data.year == 2002, "population"] = (
            0 if problem == "zero" else float(problem)
        )
    with pytest.raises(ValueError):
        add_population_features(data)


@pytest.mark.parametrize("problem", ["gap", "duplicate", "nan", "inf", "missing"])
def test_population_trainer_rejects_invalid_data(problem):
    data = pd.read_csv(MODEL_DATA_PATH)
    if problem == "gap":
        data = data.iloc[1:]
    elif problem == "duplicate":
        data.loc[1, "year"] = 2003
    elif problem == "missing":
        data = data.drop(columns="population_growth_lag1")
    else:
        data.loc[0, "population_growth_lag1"] = float(problem)
    with pytest.raises(ValueError):
        validate_model_data(data)


def test_selection_remains_validation_only(tmp_path):
    result = train_model()
    assert result["selected_model"] == "baseline"
    assert (
        min(result["comparison"], key=lambda row: row["test_mae"])["model"]
        == "population_regression"
    )
    assert result["splits"]["train"]["row_count"] == 16
    assert result["splits"]["evaluation_fit"]["row_count"] == 19
    assert result["splits"]["test"]["years"] == [2022, 2023, 2024, 2025]
    data = pd.read_csv(MODEL_DATA_PATH)
    data.loc[data.year >= 2022, "average_rent"] += 10000
    data.to_csv(tmp_path / "changed.csv", index=False)
    changed = train_model(tmp_path / "changed.csv")
    assert changed["selected_model"] == result["selected_model"]
    assert changed["evaluation_parameters"] == result["evaluation_parameters"]
    assert len(result["test_predictions"]) == 12


def test_population_outputs_reproduce_and_import_is_quiet(tmp_path):
    build_population_model_data(output_path=tmp_path / "population.csv")
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "population.csv"), pd.read_csv(MODEL_DATA_PATH)
    )
    result = train_model()
    save_results(result, tmp_path / "results")
    for filename in ("model_comparison.csv", "test_predictions.csv"):
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / "results" / filename),
            pd.read_csv(OUTPUT_DIR / filename),
            check_exact=False,
            rtol=1e-10,
            atol=1e-9,
        )
    summary = json.loads((tmp_path / "results/model_summary.json").read_text())
    assert summary["selected_model"] == "baseline"
    assert summary["splits"]["train"]["years"] == list(range(2003, 2019))
    for model in summary["evaluation_parameters"].values():
        if model["coefficients"] is not None:
            assert np.isfinite(list(model["coefficients"].values())).all()
    before = {p: p.stat().st_mtime_ns for p in OUTPUT_DIR.iterdir()}
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    process = subprocess.run(
        [sys.executable, "-c", "import models.train_population_models"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert process.stdout == ""
    assert before == {p: p.stat().st_mtime_ns for p in OUTPUT_DIR.iterdir()}
