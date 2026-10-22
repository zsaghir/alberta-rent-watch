import sqlite3
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "database" / "rent_watch.sqlite"
def inspect_schema():
    """
    read the database’s table and column information
    """
    try:
        connection = sqlite3.connect(DATABASE_PATH.as_uri()+'?more=ro',uri=True)
        cursor = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
        tables = cursor.fetchall()
        return tables
    except sqlite3.Error as e:
        print(f"An error occurred while inspecting the schema: {e}")
        return []
    finally:
        if 'connection' in locals():
            connection.close()