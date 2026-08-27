import pandas as pd

df = pd.read_csv("data/processed/edmonton_2br_master.csv")
df = df.sort_values("year").reset_index(drop=True)
df["rent_lag1"] = df["average_rent"].shift(1)
df["vacancy_lag1"] = df["vacancy_rate"].shift(1)
model_df = df.dropna(
    subset=[
        "rent_lag1",
        "vacancy_lag1"
    ]
).copy()
model_df.to_csv(
    "data/model/edmonton_2br_model.csv",
    index=False
)
print(model_df.head())
print(f"Rows: {len(model_df)}")
