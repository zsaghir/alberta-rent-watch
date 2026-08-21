import json
from pathlib import Path

import streamlit as st

DATA_PATH = Path(__file__).parent / (
    "src/alberta_rent_watch/data/processed/dashboard.json"
)

st.set_page_config(page_title="Alberta Rent Watch", page_icon=":house:")
st.title("Alberta Rent Watch")
st.caption("Edmonton two-bedroom purpose-built rental market")

try:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    records = payload["records"]
    years = [record["year"] for record in records]
    city = payload["metadata"]["geography"].split(",")[0]
except (OSError, KeyError, TypeError, ValueError) as error:
    st.error(f"Dashboard data could not be loaded: {error}")
    st.stop()
else:
    st.write(
        f"{len(records)} annual {city} observations loaded "
        f"({min(years)}–{max(years)})"
    )
