"""Export the existing dashboard data to its relocated website path."""

from pathlib import Path

from alberta_rent_watch.export import export_dashboard_data

PROJECT_ROOT = Path(__file__).parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "website" / "edmonton_2br_history.json"


if __name__ == "__main__":
    export_dashboard_data(RAW_DIR, OUTPUT_PATH)
