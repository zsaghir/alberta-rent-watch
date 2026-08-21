import polars as pl

from .cmhc import TARGET_CITY


def derive_current_rent_summary(data: pl.DataFrame) -> dict[str, object]:
    #Filter and sort the observations
    ordered = data.filter(pl.col("city") == TARGET_CITY).sort(["city", "year"])
    #Get the latest observation
    latest = ordered.tail(1).to_dicts()[0]
    #Get the rent by year
    rent_by_year = dict(ordered.select("year", "average_rent").iter_rows())
    #Get the latest year and rent
    latest_year = latest["year"]
    latest_rent = latest["average_rent"]

    return {
        "latest_year": latest_year,
        "latest_rent": latest_rent,
        "change_from_prior_year": latest_rent - rent_by_year[latest_year - 1],
        "change_since_2020": latest_rent - rent_by_year[2020],
        "change_since_2000": latest_rent - rent_by_year[2000],
    }