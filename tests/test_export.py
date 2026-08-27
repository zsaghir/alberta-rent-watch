import json
from pathlib import Path

from alberta_rent_watch.export import export_dashboard_data

RAW_DIR = Path(__file__).parents[1] / "data/raw"
FIELDS = {"year", "city", "average_rent", "vacancy_rate"}


def test_exports_verified_dashboard_records(tmp_path: Path) -> None:
    output = tmp_path / "dashboard.json"
    export_dashboard_data(RAW_DIR, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    records = payload["records"]
    assert len(records) == 26
    assert [record["year"] for record in records] == list(range(2000, 2026))
    assert len({(record["city"], record["year"]) for record in records}) == 26
    assert all(set(record) == FIELDS for record in records)
    assert tuple(records[0].values()) == (2000, "Edmonton, Alberta", 603.0, 1.3)
    assert tuple(records[-1].values()) == (2025, "Edmonton, Alberta", 1598.0, 3.8)
    assert all(record["average_rent"] is not None and record["vacancy_rate"] is not None for record in records)
    metadata = payload["metadata"]
    assert (metadata["year_start"], metadata["year_end"], metadata["average_rent_unit"], metadata["vacancy_rate_unit"]) == (2000, 2025, "CAD per month", "percent")
