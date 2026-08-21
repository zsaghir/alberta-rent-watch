import json
from datetime import datetime, timezone
from pathlib import Path

from . import cmhc


def export_dashboard_data(raw_dir: Path, output_path: Path) -> None:
    records = cmhc.load_cmhc_data(raw_dir).to_dicts()
    metadata = {
        "source": "CMHC tables distributed through Statistics Canada",
        "rent_table": cmhc.RENT_TABLE, "vacancy_table": cmhc.VACANCY_TABLE,
        "geography": cmhc.TARGET_CITY, "frequency": "annual",
        "unit_type": cmhc.TARGET_UNIT, "structure_type": cmhc.TARGET_STRUCTURE,
        "year_start": cmhc.FIRST_YEAR, "year_end": cmhc.LAST_YEAR,
        "average_rent_unit": "CAD per month", "vacancy_rate_unit": "percent",
        "generated_date": datetime.now(timezone.utc).date().isoformat(),
    }
    payload = {"metadata": metadata, "records": records}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
