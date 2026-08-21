import json
from pathlib import Path

import streamlit as st

DATA_PATH = Path(__file__).parent / (
    "src/alberta_rent_watch/data/processed/dashboard.json"
)


def format_change(value: float) -> str:
    return f"{value:+,.0f}".replace("+", "+$").replace("-", "-$")


st.set_page_config(page_title="Alberta Rent Watch", page_icon=":house:")
st.title("Alberta Rent Watch")
st.caption("Edmonton two-bedroom purpose-built rental market")

try:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    records = payload["records"]
    years = [record["year"] for record in records]
    city = payload["metadata"]["geography"].split(",")[0]
    summary = payload["rent_summary"]
except (OSError, KeyError, TypeError, ValueError) as error:
    st.error(f"Dashboard data could not be loaded: {error}")
    st.stop()
else:
    st.write(
        f"{len(records)} annual {city} observations loaded "
        f"({min(years)}–{max(years)})"
    )
    st.header(f"{city} Rental Market")
    st.subheader("Two-bedroom purpose-built rentals")
    st.metric(f"{summary['latest_year']} average monthly rent",
              f"${summary['latest_rent']:,.0f} per month")
    st.write(
        f"**Change from {summary['latest_year'] - 1}:** "
        f"{format_change(summary['change_from_prior_year'])} per month"
    )
    st.write(
        f"**Change since 2020:** "
        f"{format_change(summary['change_since_2020'])} per month"
    )
    st.write(
        f"**Change since 2000:** "
        f"{format_change(summary['change_since_2000'])} per month"
    )
