import sqlite3
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "database" / "rent_watch.sqlite"
# Restrict the tables and functions available to queries.
ALLOWED_TABLES = {
    "rental_history",
    "model_runs",
    "model_predictions",
    "forecasts",
    "dataset_metadata",
}
ALLOWED_FUNCTIONS = {"count", "avg", "min", "max", "sum", "round", "lag"}


def authorize(action, arg1, arg2, database_name, source):
    """SQLite calls this once for each operation needed by the statement."""
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ and arg1 in ALLOWED_TABLES:
        # COUNT(*) can read a table without naming a column. Some SQLite
        # versions omit the database name for that read. This fresh connection
        # cannot create temp tables or attach another database.
        table_only_read = database_name is None and arg2 == ""
        if database_name == "main" or table_only_read:
            return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in ALLOWED_FUNCTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def run_sql(query):
    """
    Read up to ten result rows from the Rent Watch database.
    """
    connection = None
    try:
        connection = sqlite3.connect(DATABASE_PATH.as_uri() + "?mode=ro", uri=True)
        connection.set_authorizer(authorize)
        deadline = time.monotonic() + 2  # Two seconds from now.

        def time_is_up():
            return time.monotonic() >= deadline

        connection.set_progress_handler(time_is_up, 1_000)
        cursor = connection.execute(query)
        columns = [field[0] for field in cursor.description]
        rows = cursor.fetchmany(11)
        return {"columns": columns, "rows": rows[:10], "truncated": len(rows) > 10}

    except sqlite3.Error as e:
        raise RuntimeError(f"SQL query failed: {e}") from e
    finally:
        if connection is not None:
            connection.close()
