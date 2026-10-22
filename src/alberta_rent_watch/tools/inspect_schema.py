import sqlite3
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "database" / "rent_watch.sqlite"
def inspect_schema():
    """
    read the database’s table and column information
    """
    try:
        connection = sqlite3.connect(DATABASE_PATH.as_uri()+'?mode=ro',uri=True)
        cursor = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
        tables = cursor.fetchall()
        schema = {}
        for table in tables:
            table_name = table[0]
            columns_cursor = connection.execute(f"SELECT name, type FROM pragma_table_info(?) ORDER BY cid;",
                                                 (table_name,),)
            columns = columns_cursor.fetchall()
            schema[table_name] = columns
            columns_metadata_cursor = connection.execute(
                """
                select column_name,description,unit, geography, scope,coverage_start, coverage_end, notes
                from dataset_metadata
                where table_name = ?
                """,
                (table_name,),
            )
            columns_metadata = columns_metadata_cursor.fetchall()
            schema[table_name] = {
                "columns": columns,
                "metadata": columns_metadata,
            }   
        
        return schema

    except Exception as e:
        raise RuntimeError(f"Unexpected error occurred: {e}")
    finally:
        if 'connection' in locals():
            connection.close()