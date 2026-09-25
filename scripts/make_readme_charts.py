"""Render README figures from the project's own data and model code."""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models.train_model import train_model

OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED = "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.8, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "axes.axisbelow": True, "legend.labelcolor": INK2,
})

master = pd.read_csv(ROOT / "data/processed/edmonton_2br_master.csv")
model_data = pd.read_csv(ROOT / "data/model/edmonton_2br_model.csv")
results = train_model()
preds = pd.DataFrame(results["test_predictions"])
test = model_data[model_data["year"] >= 2022].reset_index(drop=True)
preds["baseline"] = test["rent_lag1"].to_numpy()
preds["baseline_error"] = (preds["average_rent"] - preds["baseline"]).abs()


def dollars(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


# 1. Rent history + 2026 forecast
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(master["year"], master["average_rent"], color=BLUE, lw=2,
        marker="o", ms=4.5, label="Observed average rent")
fy, fr = results["forecast_year"], results["forecast_rent"]
last = master.iloc[-1]
ax.plot([last["year"], fy], [last["average_rent"], fr], color=BLUE, lw=2, ls=":")
ax.plot(fy, fr, marker="o", ms=9, mfc=SURFACE, mec=BLUE, mew=2,
        ls="none", label=f"{fy} model forecast")
for year, rent, dy in [(2000, 603, 14), (2008, 1037, 14), (2016, 1232, 14),
                       ]:
    ax.annotate(f"{year}: ${rent:,.0f}", (year, rent), textcoords="offset points",
                xytext=(0, dy), ha="center", color=INK2, fontsize=9)
ax.annotate(f"{fy} forecast: ${fr:,.0f}", (fy, fr), textcoords="offset points",
            xytext=(-8, 10), ha="right", color=INK, fontsize=9, fontweight="bold")
ax.annotate("2025: $1,598", (2025, 1598), textcoords="offset points",
            xytext=(8, -16), ha="left", color=INK2, fontsize=9)
ax.set_ylim(550, 1760)
ax.set_title("Edmonton two-bedroom average monthly rent, 2000–2025 (+2026 forecast)")
ax.set_xlabel("Year")
ax.set_ylabel("Average monthly rent (CAD)")
dollars(ax)
ax.set_xlim(1999, 2027.5)
ax.legend(loc="upper left")
save(fig, "01_rent_history_forecast.png")

# 2. Vacancy history
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(master["year"], master["vacancy_rate"], color=BLUE, lw=2, marker="o", ms=4.5)
for year in (2001, 2016, 2021, 2023):
    v = master.loc[master["year"] == year, "vacancy_rate"].item()
    ax.annotate(f"{v:.1f}%", (year, v), textcoords="offset points",
                xytext=(0, 8 if v > 3 else -14), ha="center", color=INK2, fontsize=9)
ax.set_title("Edmonton purpose-built rental vacancy rate, 2000–2025")
ax.set_xlabel("Year")
ax.set_ylabel("Vacancy rate (%)")
ax.set_ylim(0, 8)
save(fig, "02_vacancy_history.png")

# 3. Year-over-year rent change (diverging: increases blue, decreases red)
yoy = master.assign(change=master["average_rent"].diff()).dropna()
fig, ax = plt.subplots(figsize=(10, 3.8))
colors = [BLUE if c >= 0 else RED for c in yoy["change"]]
ax.bar(yoy["year"], yoy["change"], color=colors, width=0.72,
       edgecolor=SURFACE, linewidth=1)
ax.axhline(0, color=MUTED, lw=1)
for year in (2007, 2009, 2016, 2024, 2025):
    c = yoy.loc[yoy["year"] == year, "change"].item()
    ax.annotate(f"{c:+,.0f}", (year, c), textcoords="offset points",
                xytext=(0, 4 if c >= 0 else -13), ha="center", color=INK2, fontsize=9)
ax.set_title("Year-over-year change in average two-bedroom rent (CAD/month)")
ax.set_xlabel("Year")
ax.set_ylabel("Change from prior year")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda x, _: f"{'+' if x > 0 else '−' if x < 0 else ''}${abs(x):,.0f}"))
save(fig, "03_rent_yoy_change.png")

# 4. Held-out test: actual vs predicted
context = master[master["year"] >= 2015]
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.axvspan(2021.5, 2025.5, color="#f0efec", zorder=0)
ax.text(2021.65, 1640, "Held-out test years\n(never used to pick the model)",
        color=INK2, fontsize=9, va="top")
ax.plot(context["year"], context["average_rent"], color=BLUE, lw=2, marker="o",
        ms=5, label="Actual rent")
ax.plot(preds["year"], preds["predicted_rent"], color=ORANGE, lw=2, marker="s",
        ms=6, ls="--", label="Linear regression (selected)")
ax.plot(preds["year"], preds["baseline"], color=AQUA, lw=2, marker="^", ms=6,
        ls=":", label="Naive baseline (last year's rent)")
end = preds.iloc[-1]
ax.annotate(f"Actual ${end['average_rent']:,.0f}", (2025, end["average_rent"]),
            xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
ax.annotate(f"Regression ${end['predicted_rent']:,.0f}", (2025, end["predicted_rent"]),
            xytext=(8, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
ax.annotate(f"Baseline ${end['baseline']:,.0f}", (2025, end["baseline"]),
            xytext=(8, -4), textcoords="offset points", va="center", color=INK2, fontsize=9)
ax.set_xlim(2014.5, 2027.3)
ax.set_ylim(1150, 1660)
ax.set_title("Test-period accuracy: predicted vs actual rent, 2022–2025")
ax.set_xlabel("Year")
ax.set_ylabel("Average monthly rent (CAD)")
dollars(ax)
ax.legend(loc="upper left")
save(fig, "04_test_actual_vs_predicted.png")

# 5. Absolute error per test year
fig, ax = plt.subplots(figsize=(10, 3.8))
w = 0.36
x = preds["year"]
b1 = ax.bar(x - w / 2 - 0.01, preds["absolute_error"], w, color=ORANGE,
            label="Linear regression (selected)")
b2 = ax.bar(x + w / 2 + 0.01, preds["baseline_error"], w, color=AQUA,
            hatch="//", edgecolor=SURFACE, label="Naive baseline")
for bars in (b1, b2):
    for bar in bars:
        ax.annotate(f"${bar.get_height():,.0f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    color=INK2, fontsize=9)
ax.set_xticks(x)
ax.set_title("Absolute forecast error by test year (lower is better)")
ax.set_xlabel("Year")
ax.set_ylabel("Absolute error (CAD/month)")
dollars(ax)
ax.legend(loc="upper left")
save(fig, "05_test_absolute_error.png")


# 6. Context series collected but not yet modelled
def read_statcan_row(path, label):
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    header = next(r for r in rows if len(r) > 2 and r[0].startswith("Geography"))
    values = next(r for r in rows if r and r[0] == label)
    return pd.DataFrame({
        "year": [int(y) for y in header[1:] if y.strip().isdigit()],
        "value": [float(v.replace(",", "")) for v in values[1:len(header)] if v],
    })


pop = read_statcan_row(ROOT / "data/raw/statcan_population.csv", "Edmonton (CMA), Alberta")
unemp = read_statcan_row(ROOT / "data/raw/statcan_unemployment.csv", "Edmonton, Alberta")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
a1.plot(pop["year"], pop["value"] / 1e6, color=BLUE, lw=2, marker="o", ms=3.5)
a1.set_title("Edmonton CMA population (millions)", fontsize=11)
a1.annotate(f"{pop['value'].iloc[0] / 1e6:.2f}M", (pop["year"].iloc[0], pop["value"].iloc[0] / 1e6),
            xytext=(8, -4), textcoords="offset points", color=INK2, fontsize=9)
a1.annotate(f"{pop['value'].iloc[-1] / 1e6:.2f}M", (pop["year"].iloc[-1], pop["value"].iloc[-1] / 1e6),
            xytext=(-6, 6), textcoords="offset points", ha="right", color=INK2, fontsize=9)
a2.plot(unemp["year"], unemp["value"], color=BLUE, lw=2, marker="o", ms=3.5)
a2.set_title("Edmonton unemployment rate (%)", fontsize=11)
a2.annotate("11.9% (2020)", (2020, 11.9), xytext=(6, -2), textcoords="offset points",
            color=INK2, fontsize=9)
a2.set_ylim(0, 13.5)
for a in (a1, a2):
    a.set_xlabel("Year")
save(fig, "06_context_population_unemployment.png")

print("wrote", sorted(p.name for p in OUT.iterdir()))
