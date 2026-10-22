# Your first Rent Watch database

The database is `data/database/rent_watch.sqlite`: one local file, not a server. Python includes SQLite, so no new dependency is needed.

Rebuild it from your existing CSV and JSON outputs:

```bash
uv run python src/data/build_database.py
```

This replaces the generated database after a successful import. Keep the CSV and JSON files as the source inputs. Manual database edits will be lost on rebuild.

## Tables and relationships

| Table | Current rows | Purpose |
|---|---:|---|
| rental_history | 26 | Observed annual rent, vacancy, population and unemployment |
| model_runs | 9 | One saved candidate per experiment, with features, periods and metrics |
| model_predictions | 36 | Four historical test predictions per run |
| forecasts | 1 | The saved 2026 forecast, with its own fit period |
| dataset_metadata | 9 | Four source/column descriptions and five table descriptions |

A **primary key** uniquely identifies a row. For example, rental_history uses year.

A **foreign key** links tables. model_predictions.run_id points to model_runs.run_id. The database builder enables foreign-key checks to reject predictions linked to nonexistent runs.

Run IDs include their experiment: `economic:baseline` and `population:baseline` are different runs. The original result files contain only the selected model, so other original candidates are not invented.

`selected = 1` means selected within that experiment, not globally best. `features_json` is a JSON list stored as text. Training and evaluation periods in model_runs describe historical evaluation. The forecast has separate fit years because its model was refitted through 2025.

`NULL` means unavailable, not zero: population is missing in 2000; unemployment is missing in 2000–2010. Metadata describes units, demographic scope and sources. Table-level metadata uses `*` in column_name.

## Explore it yourself

From the repository root, run `uv run python`, then enter:

```python
import sqlite3
from pathlib import Path

path = Path('data/database/rent_watch.sqlite').resolve()
connection = sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True)

# execute() runs SQL; fetchall() returns the result rows to Python.
print(connection.execute('''
    SELECT year, average_rent, vacancy_rate
    FROM rental_history
    ORDER BY year DESC
    LIMIT 5
''').fetchall())
```

List the tables and inspect the history columns:

```python
print(connection.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall())
print(connection.execute('PRAGMA table_info(rental_history)').fetchall())
```

Join predictions to their experiment:

```python
print(connection.execute('''
    SELECT r.experiment, r.model_name, p.year,
           p.actual_rent, p.predicted_rent
    FROM model_predictions AS p
    JOIN model_runs AS r ON r.run_id = p.run_id
    WHERE r.experiment = 'population'
    ORDER BY r.model_name, p.year
''').fetchall())
connection.close()
```

These are manual learning queries, not an agent tool. Opening with mode=ro prevents writes on that connection. A future LLM SQL tool will still need query validation, allowed-operation restrictions, and execution limits.

## How the builder works

1. Read and validate the master CSV.
2. Create a temporary database and its five tables.
3. Insert observations and saved results using parameter placeholders (`?`).
4. Add source and interpretation notes.
5. Check foreign keys and database integrity.
6. Replace the generated database only after a successful build.

The builder does not retrain models, generate new forecasts, call an LLM, or implement inspect_schema/run_sql. Those tools remain your next learning steps.
