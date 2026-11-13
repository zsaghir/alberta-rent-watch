import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "database" / "rent_watch.sqlite"


def inspect_schema():
    """
    read the database’s table and column information
    """
    connection = None
    try:
        connection = sqlite3.connect(DATABASE_PATH.as_uri() + "?mode=ro", uri=True)
        cursor = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        )
        tables = cursor.fetchall()
        schema = {}
        for table in tables:
            table_name = table[0]
            columns_cursor = connection.execute(
                "SELECT name, type FROM pragma_table_info(?) ORDER BY cid;",
                (table_name,),
            )
            columns = columns_cursor.fetchall()
            columns_metadata_cursor = connection.execute(
                """
                select column_name,description,unit, geography, scope,coverage_start, coverage_end, notes
                from dataset_metadata
                where table_name = ?
                """,
                (table_name,),
            )
            # Use query field names so callers do not need to know tuple positions.
            metadata_fields = [
                field[0] for field in columns_metadata_cursor.description
            ]
            columns_metadata = [
                dict(zip(metadata_fields, row))
                for row in columns_metadata_cursor.fetchall()
            ]
            schema[table_name] = {
                "columns": columns,
                "metadata": columns_metadata,
            }

        return schema

    except Exception as e:
        raise RuntimeError(f"Unexpected error occurred: {e}") from e
    finally:
        if connection is not None:
            connection.close()
