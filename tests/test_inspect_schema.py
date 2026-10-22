"""Check the schema tool without changing the real database."""

import importlib
import sqlite3

import pytest

from data.build_database import build_database

# Import the module so monkeypatch can temporarily change its DATABASE_PATH.
schema_module = importlib.import_module("alberta_rent_watch.tools.inspect_schema")


@pytest.fixture
def schema_database(tmp_path, monkeypatch):
    """Give each test a fresh database; restore the path automatically afterward."""
    database_path = build_database(output_path=tmp_path / "rent_watch.sqlite")
    monkeypatch.setattr(schema_module, "DATABASE_PATH", database_path)
    return database_path


def test_returns_expected_tables(schema_database):
    result = schema_module.inspect_schema()

    # Sets compare table names without depending on their order.
    assert set(result) == {
        "dataset_metadata",
        "forecasts",
        "model_predictions",
        "model_runs",
        "rental_history",
    }


def test_returns_columns_and_metadata(schema_database):
    result = schema_module.inspect_schema()

    assert ("average_rent", "REAL") in result["rental_history"]["columns"]
    for table in result.values():
        assert set(table) == {"columns", "metadata"}
        assert table["columns"]
        assert table["metadata"]

    # Metadata tuple positions follow the SELECT columns in your tool.
    rent_metadata = next(
        row for row in result["rental_history"]["metadata"] if row[0] == "average_rent"
    )
    assert rent_metadata[2] == "CAD/month"
    assert rent_metadata[3] == "Edmonton CMA, Alberta"
    assert rent_metadata[5:7] == (2000, 2025)


def test_missing_database_raises_without_creating_file(tmp_path, monkeypatch):
    missing_path = tmp_path / "missing.sqlite"
    monkeypatch.setattr(schema_module, "DATABASE_PATH", missing_path)

    # Your function wraps SQLite's connection error in a RuntimeError.
    with pytest.raises(RuntimeError, match="unable to open database file") as error:
        schema_module.inspect_schema()

    assert isinstance(error.value.__context__, sqlite3.OperationalError)
    assert not missing_path.exists()


def test_inspection_leaves_database_unchanged(schema_database):
    before = schema_database.read_bytes()

    schema_module.inspect_schema()

    assert schema_database.read_bytes() == before
