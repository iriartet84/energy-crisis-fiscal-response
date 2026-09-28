# Energy Shocks and Fiscal Policy

Code for my Master's thesis, which asks whether institutional quality shapes how governments' fiscal positions respond to exogenous oil supply shocks.

The analysis uses a panel of 30 countries over 1975–2024 and combines several ingredients:

- the oil supply news shock of Känzig
- fiscal data from the IMF and World Bank
- institutional indicators from V-Dem

The main tool is a set of state-dependent local projections, where the response of the primary balance to the shock depends on a country's institutional quality.

## Files

| File | What it does |
|---|---|
| `data_prep.py` | Loads the panel from `full-panel-data.db` and builds the estimation variables. It also produces the descriptive tables and the institutional-quality and oil-shock figures. |
| `lp_analysis.py` | Estimates the local projections and loads the bootstrap results. It writes all regression tables and figures. |
| `full-panel-data.db` | Finished country-year panel (SQLite, table `panel`) |
| `bootstrapped-data/` | Saved point estimates and bootstrap draws (joblib pickles) |
| `outputs/` | Tables (CSV) and figures (PNG) produced by the two scripts |

The panel in `full-panel-data.db` was built from the raw sources, which are not included in this repository:

- IMF: primary balance, expenditure, revenue, debt.
- World Bank WDI: GDP, CPI, tax revenue, deflator, energy imports.
- World Bank commodity prices: Brent.
- V-Dem: institutional indicators.
- Känzig: oil supply news shock.
- Caldara and Iacoviello: geopolitical risk.

## Method

**Institutional quality.** Six V-Dem measures are z-scored and averaged:

- control of corruption
- rule of law
- property rights
- transparent laws
- bureaucratic quality
- electoral democracy

The average is then standardised again to give the index `inst`.

**Outcome.** The cumulative change in the primary balance (% of GDP) between t and t+h, for h = 0, ..., 10.

**Smooth-transition LP (main specification).** The shock is interacted with a logistic function of lagged institutions:

`F = 1 / (1 + exp(-lambda * (inst_lag1 - c)))`

- The centre c and speed lambda are chosen by grid search over the SSR summed over h = 0–4.
- The regression includes country fixed effects and two lags of GDP and CPI inflation.

**Binary threshold LP.** The same regression, split at a threshold on each country's mean institutional quality. The threshold is found by grid search.

**Robustness.** The same models re-estimated with:

- an extended control set (geopolitical risk, energy imports, world GDP, oil share)
- a linear trend
- lags of the primary balance and the shock
- energy-import exposure interactions

**Decomposition.** The binary LP for revenue, tax revenue, expenditure and debt, both in % of GDP and in real terms (logs, and the inverse hyperbolic sine for the real primary balance).

**Sub-indices.** The smooth-transition LP run separately on each V-Dem component.

**Inference.** Country block bootstrap: countries are resampled with replacement using seed 42 + b, and the grid search is repeated in every draw.

## Running it

```bash
pip install -r requirements.txt
python data_prep.py
python lp_analysis.py
```

`lp_analysis.py` re-estimates every point estimate from the panel, including all grid searches, and takes about 3–4 minutes. The bootstrap draws are loaded from `bootstrapped-data/` because rerunning them takes hours.

To rerun the bootstraps, set `RERUN_BOOTSTRAP = True` at the top of `lp_analysis.py`. New draws go to `outputs/bootstrap_rerun/` and the saved files are left untouched. The sub-index bootstraps are the exception: they always use the saved draws.

`outputs/tables/replication_check.csv` compares the re-estimated point estimates with the saved ones. All differences are below 1e-3, and most are around 1e-5. The small gaps come from some saved files being produced on a slightly earlier version of the panel.

## Main result

- **Transition.** c = −0.943 and lambda = 59.3, so the switch between states is very sharp: F moves from 0.5 to 0.9 within 0.037 units of the index.
- **Response at h = 4.**
    - Low-institution state: the primary balance falls by 0.185 pp of GDP after a one-unit shock.
    - High-institution state: it rises by 0.157 pp.

## Notes

- The `gdp_growth` and `world_gdp_growth` columns in the panel contain GDP levels (current US$), not growth rates, because of the World Bank series that was downloaded. They are used as stored.
- Mexico and Singapore have no V-Dem or fiscal data in the panel, and Egypt has no primary balance data. The regressions therefore use 30 of the 33 countries.
- `panel_annual.parquet` is a processed copy of the panel. `data_prep.py` rebuilds it from the database and checks that the two match.
