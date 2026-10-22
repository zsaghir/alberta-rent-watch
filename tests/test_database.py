import sqlite3

import pytest

from data.build_database import build_database


def test_database_contents(tmp_path):
    path = build_database(output_path=tmp_path / "rent.sqlite")
    with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as con:
        assert con.execute(
            "SELECT * FROM rental_history WHERE year=2025"
        ).fetchone() == (2025, 1598.0, 3.8, 1692385, 7.7)
        for table, count in [
            ("rental_history", 26),
            ("model_runs", 9),
            ("model_predictions", 36),
            ("forecasts", 1),
            ("dataset_metadata", 9),
        ]:
            assert con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == count
        assert (
            con.execute(
                "SELECT COUNT(*) FROM rental_history WHERE population IS NULL"
            ).fetchone()[0]
            == 1
        )
        assert (
            con.execute(
                "SELECT COUNT(*) FROM rental_history WHERE unemployment_rate IS NULL"
            ).fetchone()[0]
            == 11
        )
        assert con.execute(
            "SELECT year,predicted_rent,fit_end FROM forecasts"
        ).fetchall() == [(2026, 1657.29, 2025)]
        assert con.execute(
            "SELECT experiment,train_start FROM model_runs WHERE selected=1 ORDER BY experiment"
        ).fetchall() == [("economic", 2012), ("original", 2001), ("population", 2003)]
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []
        assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            con.execute("DELETE FROM rental_history")


def test_rebuild_and_foreign_keys(tmp_path):
    path = tmp_path / "rent.sqlite"
    build_database(output_path=path)
    build_database(output_path=path)
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0] == 9
        con.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("INSERT INTO model_predictions VALUES ('unknown',2025,1,1,0)")


def test_failed_build_preserves_existing_database(tmp_path, monkeypatch):
    import data.build_database as module

    path = build_database(output_path=tmp_path / "rent.sqlite")
    original = path.read_bytes()

    def fail(*args):
        raise ValueError("Invalid source")

    monkeypatch.setattr(module, "load_models", fail)
    with pytest.raises(ValueError, match="Invalid source"):
        build_database(output_path=path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
