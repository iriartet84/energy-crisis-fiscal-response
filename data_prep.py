import os
import sqlite3
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", message="adfuller currently returns")

base_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(base_dir, "full-panel-data.db")
cache_dir = os.path.join(base_dir, "bootstrapped-data")
table_dir = os.path.join(base_dir, "outputs", "tables")
figure_dir = os.path.join(base_dir, "outputs", "figures")

# V-Dem components of the institutional index (already z-scored in the panel)
inst_components = ["control_of_corruption", "rule_of_law", "property_rights",
                   "transparent_laws", "bureaucratic_quality", "democracy"]

inst_labels = {
    "control_of_corruption": "Control of corruption",
    "rule_of_law": "Rule of law",
    "property_rights": "Property rights",
    "transparent_laws": "Transparent laws",
    "bureaucratic_quality": "Bureaucratic quality",
    "democracy": "Democracy",
    "inst": "Composite index",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_panel():

    # Load the panel data

    conn = sqlite3.connect(db_path)
    panel = pd.read_sql("SELECT * FROM panel", conn)
    conn.close()

    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["country", "date"]).reset_index(drop=True)

    # State capacity is empty in the V-Dem extract
    panel["state_capacity"] = pd.to_numeric(panel["state_capacity"])
    panel["v3ststeecap"] = pd.to_numeric(panel["v3ststeecap"])

    # Construct the institutional index

    panel["inst"] = panel[inst_components].mean(axis=1)
    panel["inst"] = (panel["inst"] - panel["inst"].mean()) / panel["inst"].std()

    # Country mean over 1974-2024 for the binary regimes
    window = panel[panel["year"].between(1974, 2024)]
    panel["inst_mean"] = panel["country"].map(window.groupby("country")["inst"].mean())

    # Lagged institutions for the smooth transition
    panel["inst_lag1"] = panel.groupby("country")["inst"].shift(1)

    # Impute missing energy imports with the country mean

    country_mean = panel.groupby("country")["energy_imports"].transform("mean")
    panel["energy_imports"] = panel["energy_imports"].fillna(country_mean)

    panel["energy_imports_centered"] = panel["energy_imports"] - panel["energy_imports"].mean()
    panel["shock_weighted"] = panel["shock"] * panel["energy_imports"]
    panel["shock_std"] = panel["shock"] / panel["shock"].std()
    panel["imports_centered"] = panel["energy_imports"] - panel["energy_imports"].mean()

    # Real fiscal levels and log / IHS transformations

    floor = 1e-6
    panel["real_expenditure"] = panel["expenditure"] / 100 * panel["real_gdp"]
    panel["real_revenue"] = panel["gov_revenue"] / 100 * panel["real_gdp"]
    panel["real_debt"] = panel["debt"] / 100 * panel["real_gdp"]
    panel["real_pb"] = panel["pb"] / 100 * panel["real_gdp"]

    panel["log_expenditure"] = np.log(panel["real_expenditure"].clip(lower=floor))
    panel["log_debt"] = np.log(panel["real_debt"].clip(lower=floor))
    panel["log_real_tax"] = np.log(panel["tax_real"].clip(lower=floor))
    panel["log_real_gov_rev"] = np.log(panel["real_revenue"].clip(lower=floor))
    panel["ihs_pb"] = np.arcsinh(panel["real_pb"])

    # Linear trend for the robustness check
    panel["trend"] = panel["year"] - 2000

    return panel


def compare_with_parquet(panel):

    # Check the rebuilt panel against the processed copy

    path = os.path.join(cache_dir, "panel_annual.parquet")
    if not os.path.exists(path):
        return None

    stored = pd.read_parquet(path)
    cols = [c for c in stored.columns if c not in ["date", "country"] and c in panel.columns]

    a = panel[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    b = stored[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    same_missing = (np.isnan(a) == np.isnan(b)).all()
    max_diff = np.nanmax(np.abs(a - b))
    return same_missing, max_diff


def estimation_sample(panel):

    # Rows that enter the baseline regression at h = 0

    df = panel.copy()
    lag_cols = []
    for v in ["gdp_growth", "cpi_infl"]:
        for lag in [1, 2]:
            df[f"{v}_lag{lag}"] = df.groupby("country")[v].shift(lag)
            lag_cols.append(f"{v}_lag{lag}")

    return df.dropna(subset=["pb", "shock", "inst_lag1", "inst_mean"] + lag_cols)


def adf_table(panel):

    # Share of countries where the ADF test rejects a unit root at 5%

    variables = {
        "pb": "Primary balance",
        "expenditure": "Government expenditure",
        "gov_revenue": "Government revenue",
        "debt": "Public debt",
        "gdp_growth": "GDP growth",
        "cpi_infl": "CPI inflation (YoY)",
        "gpr": "Geopolitical risk (GPR)",
        "energy_imports": "Energy imports",
        "world_gdp_growth": "World GDP growth",
    }

    rows = []
    for var, label in variables.items():
        p_level, p_diff = [], []
        for _, grp in panel.groupby("country"):
            s = grp.set_index("date")[var]
            if s.dropna().shape[0] >= 15:
                p_level.append(adfuller(s.dropna(), maxlag=4, autolag="AIC", regression="c")[1])
            if s.diff().dropna().shape[0] >= 15:
                p_diff.append(adfuller(s.diff().dropna(), maxlag=4, autolag="AIC", regression="c")[1])

        rows.append({"variable": label,
                     "share_stationary_level": np.mean(np.array(p_level) < 0.05),
                     "share_stationary_first_diff": np.mean(np.array(p_diff) < 0.05)})

    # The oil shock is common to all countries, so it is tested once
    shock = panel.dropna(subset=["shock"]).groupby("year")["shock"].mean()
    p = adfuller(shock, maxlag=4, autolag="AIC", regression="c")[1]
    rows.append({"variable": "Oil supply shock (Känzig)", "share_stationary_level": float(p < 0.05),
                 "share_stationary_first_diff": np.nan})

    return pd.DataFrame(rows)


def plot_institutions(panel, best_gamma):

    # Country ranking by mean institutional quality

    country_inst = panel.groupby("country")["inst_mean"].first().dropna().sort_values()
    colors = ["#c0392b" if x <= best_gamma else "#2980b9" for x in country_inst]

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(country_inst.index, country_inst.values, color=colors, height=0.7)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.axvline(best_gamma, color="black", linestyle="--", linewidth=1.2,
               label=f"Binary threshold = {best_gamma:.2f}")

    for y, x in enumerate(country_inst.values):
        ax.text(x + (0.03 if x > 0 else -0.03), y, f"{x:.2f}", va="center",
                ha="left" if x > 0 else "right", fontsize=8)

    ax.set_xlim(country_inst.min() - 0.3, country_inst.max() + 0.3)
    ax.set_xlabel("Composite institutional quality, country mean 1974-2024 (standardised)")
    ax.set_title("Institutional quality by country")
    ax.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "institutional_index_by_country.png"), dpi=200)
    plt.close()

    # Institutional quality over time

    fig, ax = plt.subplots(figsize=(11, 5))
    for country, grp in panel.dropna(subset=["inst", "inst_mean"]).groupby("country"):
        color = "#c0392b" if grp["inst_mean"].iloc[0] <= best_gamma else "#2980b9"
        ax.plot(grp["year"], grp["inst"], color=color, alpha=0.5, linewidth=0.9)

    ax.axhline(best_gamma, color="black", linestyle="--", linewidth=1.2,
               label=f"Binary threshold = {best_gamma:.2f}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Composite institutional quality")
    ax.set_title("Institutional quality over time (red: low regime, blue: high regime)")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "institutional_index_over_time.png"), dpi=200)
    plt.close()


def plot_oil_shock(panel):

    # Annual Känzig shock against the Brent price

    annual = panel.groupby("year")[["shock", "brent"]].first().loc[1975:2024]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(annual.index, annual["shock"], color="#c0392b", alpha=0.7, width=0.8,
            label="Oil supply news shock")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Oil supply news shock (annual sum)")

    ax2 = ax1.twinx()
    ax2.plot(annual.index, annual["brent"], color="#2980b9", linewidth=2, label="Brent (USD per barrel)")
    ax2.set_ylabel("USD per barrel")
    ax2.spines["right"].set_visible(True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)
    ax1.set_title("Känzig oil supply news shock and the Brent oil price")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "oil_shock_and_brent.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    # Load the panel and construct the estimation variables

    panel = load_panel()

    check = compare_with_parquet(panel)
    if check is not None:
        print(f"Rebuilt panel vs panel_annual.parquet: same missing = {check[0]}, "
              f"max abs diff = {check[1]:.1e}")
        pd.DataFrame([{"same_missing": check[0], "max_abs_diff": check[1]}]).to_csv(
            os.path.join(table_dir, "data_consistency_check.csv"), index=False)

    # Describe the estimation sample

    sample = estimation_sample(panel)
    print(f"Panel: {panel['country'].nunique()} countries, {panel['year'].min()}-{panel['year'].max()}")
    print(f"Estimation sample: {sample['country'].nunique()} countries, "
          f"{sample['year'].min()}-{sample['year'].max()}, {len(sample)} observations")

    by_country = (sample.groupby("country")
                  .agg(first_year=("year", "min"), last_year=("year", "max"),
                       observations=("year", "size"), inst_mean=("inst_mean", "first"))
                  .reset_index())
    by_country.to_csv(os.path.join(table_dir, "sample_by_country.csv"), index=False)

    variables = ["pb", "gov_revenue", "tax_rev_gdp", "expenditure", "debt", "shock", "inst",
                 "gdp_growth", "cpi_infl", "gpr", "energy_imports", "oil_share"]
    sample[variables].describe().T.to_csv(os.path.join(table_dir, "descriptive_statistics.csv"))

    adf_table(panel).to_csv(os.path.join(table_dir, "adf_unit_root.csv"), index=False)

    # Institutional quality tables and figures

    best_gamma = joblib.load(os.path.join(cache_dir, "baseline_binary_long_parallel.pkl"))["best_gamma"]

    ranking = panel.groupby("country")["inst_mean"].first().dropna().sort_values().reset_index()
    ranking["regime"] = np.where(ranking["inst_mean"] <= best_gamma, "Low", "High")
    ranking.to_csv(os.path.join(table_dir, "institutional_index_by_country.csv"), index=False)

    corr = panel[inst_components + ["inst"]].dropna().corr().rename(index=inst_labels, columns=inst_labels)
    corr.to_csv(os.path.join(table_dir, "institutional_correlations.csv"))

    plot_institutions(panel, best_gamma)
    plot_oil_shock(panel)
