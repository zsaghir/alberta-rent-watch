"""Build a small SQLite database from validated Rent Watch outputs."""

from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
from pathlib import Path

import polars as pl

from data.build_master import validate_master

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data/database/rent_watch.sqlite"
TABLES = (
    "rental_history",
    "model_runs",
    "model_predictions",
    "forecasts",
    "dataset_metadata",
)

# Primary keys identify rows. Foreign keys link predictions to their model run.
SCHEMA = """
CREATE TABLE rental_history (
    year INTEGER PRIMARY KEY,
    average_rent REAL NOT NULL CHECK (average_rent > 0),
    vacancy_rate REAL NOT NULL CHECK (vacancy_rate BETWEEN 0 AND 100),
    population INTEGER CHECK (population > 0),
    unemployment_rate REAL CHECK (unemployment_rate BETWEEN 0 AND 100)
);
CREATE TABLE model_runs (
    run_id TEXT PRIMARY KEY,
    experiment TEXT NOT NULL,
    model_name TEXT NOT NULL,
    features_json TEXT NOT NULL,
    train_start INTEGER NOT NULL,
    train_end INTEGER NOT NULL,
    validation_start INTEGER NOT NULL,
    validation_end INTEGER NOT NULL,
    evaluation_fit_start INTEGER NOT NULL,
    evaluation_fit_end INTEGER NOT NULL,
    test_start INTEGER NOT NULL,
    test_end INTEGER NOT NULL,
    validation_mae REAL NOT NULL CHECK (validation_mae >= 0),
    test_mae REAL NOT NULL CHECK (test_mae >= 0),
    test_rmse REAL NOT NULL CHECK (test_rmse >= 0),
    test_r2 REAL NOT NULL,
    selected INTEGER NOT NULL CHECK (selected IN (0, 1)),
    source_path TEXT NOT NULL
);
CREATE TABLE model_predictions (
    run_id TEXT NOT NULL REFERENCES model_runs(run_id),
    year INTEGER NOT NULL REFERENCES rental_history(year),
    actual_rent REAL NOT NULL,
    predicted_rent REAL NOT NULL,
    absolute_error REAL NOT NULL CHECK (absolute_error >= 0),
    PRIMARY KEY (run_id, year)
);
CREATE TABLE forecasts (
    run_id TEXT NOT NULL REFERENCES model_runs(run_id),
    year INTEGER NOT NULL,
    predicted_rent REAL NOT NULL CHECK (predicted_rent > 0),
    fit_start INTEGER NOT NULL,
    fit_end INTEGER NOT NULL,
    horizon_years INTEGER NOT NULL CHECK (horizon_years = 1),
    source_path TEXT NOT NULL,
    PRIMARY KEY (run_id, year)
);
CREATE TABLE dataset_metadata (
    table_name TEXT NOT NULL,
    column_name TEXT NOT NULL,
    description TEXT NOT NULL,
    unit TEXT NOT NULL,
    geography TEXT NOT NULL,
    scope TEXT NOT NULL,
    frequency TEXT NOT NULL,
    coverage_start INTEGER,
    coverage_end INTEGER,
    source_table_id TEXT,
    source_url TEXT,
    release_date TEXT,
    notes TEXT NOT NULL,
    PRIMARY KEY (table_name, column_name)
);
"""


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def insert_run(
    connection, run_id, experiment, model, features, splits, metrics, source
):
    """Keep experiment identity and training periods attached to every model."""
    connection.execute(
        "INSERT INTO model_runs VALUES (" + ",".join(["?"] * 18) + ")",
        (
            run_id,
            experiment,
            model,
            json.dumps(features),
            min(splits["train"]),
            max(splits["train"]),
            min(splits["validation"]),
            max(splits["validation"]),
            min(splits["evaluation_fit"]),
            max(splits["evaluation_fit"]),
            min(splits["test"]),
            max(splits["test"]),
            metrics["validation_mae"],
            metrics["test_mae"],
            metrics["test_rmse"],
            metrics["test_r2"],
            int(metrics["selected"]),
            source,
        ),
    )


def load_models(connection: sqlite3.Connection, root: Path) -> None:
    """Import saved results; this does not retrain or choose new models."""
    original_path = "data/model/results/model_summary.json"
    summary = read_json(root / original_path)
    years = [
        int(row["year"]) for row in read_csv(root / "data/model/edmonton_2br_model.csv")
    ]
    model = summary["selected_model"]
    run_id = f"original:{model}"
    splits = {
        "train": [year for year in years if year <= 2018],
        "validation": [2019, 2020, 2021],
        "evaluation_fit": [year for year in years if year <= 2021],
        "test": summary["test_years"],
    }
    metrics = {
        **summary,
        "selected": True,
    }
    # Original summaries name their validation metrics with an "_mae" suffix.
    metrics["validation_mae"] = summary["validation"][
        "baseline_mae" if model == "baseline" else "linear_regression_mae"
    ]
    insert_run(
        connection,
        run_id,
        "original",
        model,
        ["rent_lag1"] if model == "baseline" else ["rent_lag1", "vacancy_lag1"],
        splits,
        metrics,
        original_path,
    )
    for row in read_csv(root / "data/model/results/test_predictions.csv"):
        insert_prediction(connection, run_id, row)
    # The forecast model was refitted through 2025, unlike the evaluation fit.
    connection.execute(
        "INSERT INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            summary["forecast"]["year"],
            summary["forecast"]["predicted_rent"],
            min(years),
            max(years),
            1,
            original_path,
        ),
    )

    for experiment in ("economic", "population"):
        relative = f"data/model/results/{experiment}/model_summary.json"
        summary = read_json(root / relative)
        splits = {name: values["years"] for name, values in summary["splits"].items()}
        for metrics in summary["comparison"]:
            model = metrics["model"]
            insert_run(
                connection,
                f"{experiment}:{model}",
                experiment,
                model,
                summary["feature_sets"][model],
                splits,
                metrics,
                relative,
            )
        for row in read_csv(
            root / f"data/model/results/{experiment}/test_predictions.csv"
        ):
            insert_prediction(connection, f"{experiment}:{row['model']}", row)


def insert_prediction(connection, run_id, row):
    connection.execute(
        "INSERT INTO model_predictions VALUES (?, ?, ?, ?, ?)",
        (
            run_id,
            int(row["year"]),
            float(row["average_rent"]),
            float(row["predicted_rent"]),
            float(row["absolute_error"]),
        ),
    )


def load_metadata(connection: sqlite3.Connection) -> None:
    """Explain units and coverage, not just SQL column names."""
    rental_scope = (
        "Two-bedroom purpose-built row/apartment rentals, structures with 3+ units"
    )
    for column, description, unit, start, table, released, scope, notes in (
        (
            "average_rent",
            "Average monthly rent",
            "CAD/month",
            2000,
            "34-10-0133-01",
            "2025-12-17",
            rental_scope,
            "Survey average, not advertised asking rent.",
        ),
        (
            "vacancy_rate",
            "Vacancy rate",
            "percent",
            2000,
            "34-10-0130-01",
            "2025-12-17",
            rental_scope,
            "Changes in rates are percentage points.",
        ),
        (
            "population",
            "Population estimate as of July 1",
            "persons",
            2001,
            "17-10-0148-01",
            "2026-01-14",
            "All ages, total gender, 2021 CMA boundaries",
            "2000 is missing. Current revised estimates; 2025 preliminary.",
        ),
        (
            "unemployment_rate",
            "Annual unemployment rate",
            "percent",
            2011,
            "14-10-0461-01",
            "2026-01-09",
            "Age 15+, total gender, 2021 CMA boundaries",
            "2000–2010 missing; do not interpret nulls as zeros.",
        ),
    ):
        connection.execute(
            "INSERT INTO dataset_metadata VALUES (" + ",".join(["?"] * 13) + ")",
            (
                "rental_history",
                column,
                description,
                unit,
                "Edmonton CMA, Alberta",
                scope,
                "annual",
                start,
                2025,
                table,
                f"https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid={table.replace('-', '')}",
                released,
                notes,
            ),
        )
    for table, description, notes in (
        (
            "rental_history",
            "Observed annual history",
            "Year means observation year, not publication year. Historical boundaries may differ. Calgary is unsupported.",
        ),
        (
            "model_runs",
            "Saved model evaluations",
            "MAE/RMSE: CAD/month; R²: unitless. Selected means selected within that experiment. Compare training periods before comparing metrics. features_json contains a JSON list.",
        ),
        (
            "model_predictions",
            "Historical test predictions",
            "Rent and error: CAD/month. Join run_id to model_runs. Test years have already been inspected; results are exploratory.",
        ),
        (
            "forecasts",
            "Saved next-year forecast",
            "CAD/month. Only 2026 is supported; 2027 is unavailable. fit_start/end describe the forecast refit, not the historical evaluation.",
        ),
        (
            "dataset_metadata",
            "Source and interpretation notes",
            "Current revised exports do not reconstruct information available at historical forecast dates.",
        ),
    ):
        connection.execute(
            "INSERT INTO dataset_metadata VALUES (" + ",".join(["?"] * 13) + ")",
            (
                table,
                "*",
                description,
                "see column notes",
                "Edmonton CMA, Alberta",
                rental_scope if table != "dataset_metadata" else "All exported data",
                "annual",
                None,
                None,
                None,
                None,
                None,
                notes,
            ),
        )


def build_database(root: Path = PROJECT_ROOT, output_path: Path = OUTPUT_PATH) -> Path:
    """Build a fresh database, replacing the old copy only after success."""
    root, output_path = Path(root), Path(output_path)
    master = pl.read_csv(root / "data/processed/edmonton_2br_master.csv")
    validate_master(master)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # A failed import must not leave a half-built database in place.
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, suffix=".sqlite", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with sqlite3.connect(temporary_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(SCHEMA)
            connection.executemany(
                "INSERT INTO rental_history VALUES (?, ?, ?, ?, ?)", master.iter_rows()
            )
            load_models(connection, root)
            load_metadata(connection)
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("Database contains invalid model references")
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Database integrity check failed")
        connection.close()
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def main() -> None:
    path = build_database()
    print(f"Created {path}")
    with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as connection:
        for table in TABLES:
            print(
                f"{table}: {connection.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]} rows"
            )


if __name__ == "__main__":
    main()
