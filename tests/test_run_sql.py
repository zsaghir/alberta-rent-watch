import importlib

import pytest

from data.build_database import build_database

tool = importlib.import_module("alberta_rent_watch.tools.run_sql")


@pytest.fixture
def run(tmp_path, monkeypatch):
    path = build_database(output_path=tmp_path / "test.sqlite")
    monkeypatch.setattr(tool, "DATABASE_PATH", path)
    return tool.run_sql


def test_reads_named_columns(run):
    assert run("SELECT year, average_rent FROM rental_history WHERE year=2025") == {
        "columns": ["year", "average_rent"],
        "rows": [(2025, 1598.0)],
        "truncated": False,
    }


def test_allows_count(run):
    assert run("SELECT COUNT(*) AS n FROM rental_history")["rows"] == [(26,)]


def test_empty_result(run):
    assert run("SELECT year FROM rental_history WHERE year=1990")["rows"] == []


def test_truncates_only_when_more_than_ten(run):
    query = "SELECT year FROM rental_history ORDER BY year"
    result = run(query)
    assert result["truncated"] and len(result["rows"]) == 10
    assert not run(query + " LIMIT 10")["truncated"]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT name FROM sqlite_master",
        "DELETE FROM rental_history",
        "SELECT upper('hello')",
        "ATTACH DATABASE ':memory:' AS other",
    ],
    ids=["catalogue", "delete", "unlisted_function", "attach"],
)
def test_denies_disallowed_operations(run, sql):
    with pytest.raises(RuntimeError, match="(authorized|prohibited)"):
        run(sql)
