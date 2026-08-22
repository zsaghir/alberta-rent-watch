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
    metadata = payload["metadata"]
    city = metadata["geography"].split(",")[0]
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
    st.subheader("Average monthly rent over time")
    chart_records = [
        {**record, "rent_change": None if index == 0 else
         record["average_rent"] - records[index - 1]["average_rent"]}
        for index, record in enumerate(records)
    ]
    st.vega_lite_chart(chart_records, {
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {"field": "year", "type": "ordinal", "title": "Year"},
            "y": {"field": "average_rent", "type": "quantitative", "title": "Average monthly rent (CAD)"},
            "tooltip": [
                {"field": "year", "type": "ordinal", "title": "Year"},
                {"field": "average_rent", "type": "quantitative", "format": "$,.0f", "title": "Average rent (CAD/month)"},
                {"field": "rent_change", "type": "quantitative", "format": "+$,.0f", "title": "Change from prior year (CAD/month)"},
                {"field": "vacancy_rate", "type": "quantitative", "format": ".1f", "title": "Vacancy rate (%)"},
            ],
        }})
    with st.expander("Accessible chart data (rent/change in CAD per month; vacancy in percent)"):
        st.dataframe(chart_records, hide_index=True)
    st.caption(
        f"Text summary: Average rent was ${records[0]['average_rent']:,.0f} "
        f"in {records[0]['year']} and ${records[-1]['average_rent']:,.0f} "
        f"in {records[-1]['year']}."
    )
    st.divider()
    st.caption(f"Source: {metadata['source']}.")
    st.caption(
        f"Scope: {metadata['geography']}; {metadata['frequency']} "
        f"{metadata['unit_type'].lower()} in "
        f"{metadata['structure_type'].lower()}, "
        f"{metadata['year_start']}–{metadata['year_end']}."
    )
