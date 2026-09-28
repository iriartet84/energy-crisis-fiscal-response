# Energy Shocks and Fiscal Policy

Replication package for the Master's thesis *Energy Shocks and Fiscal Policy*.

**Research question.** Does institutional quality shape the fiscal response to exogenous oil supply shocks?

The package reproduces the thesis results from two supplied inputs: the finished country-year panel (`full-panel-data.db`) and the completed estimation and bootstrap outputs (`bootstrapped-data/`). It does not rebuild the panel from the raw IMF, World Bank, V-Dem and oil-shock downloads.

## Methodology

- **Shock.** Känzig's oil supply news shock, summed from monthly to annual frequency. It is common to all countries.
- **Outcome.** The cumulative change in the primary balance (% of GDP), `pb(t+h) - pb(t)`, for h = 0, ..., 10. The fiscal decomposition also uses revenue, tax revenue, expenditure and debt (% of GDP), plus real-level versions: logs, and the inverse hyperbolic sine of the real primary balance, `arcsinh(pb / 100 * real_gdp)`.
- **Institutional quality.** V-Dem control of corruption (inverted `v2x_corr`), rule of law, property rights, transparent laws, bureaucratic quality and electoral democracy. Each is z-scored, then averaged, and the average is standardised again (`inst`). The binary models split on the country mean of `inst` over 1974–2024 (`inst_mean`). The smooth-transition models use the lagged index (`inst_lag1`).
- **State-dependent local projections.** These use country fixed effects (within transformation) and no time effects, since the shock is common to all countries. The horizon of interest is h = 4.
  - *Smooth transition:* `dpb(h) = a_i + bF(h) shock F(inst_lag1) + b1mF(h) shock (1 - F) + controls`, with `F = 1 / (1 + exp(-lambda (inst_lag1 - c)))`. The centre c and speed lambda are chosen by grid search to minimise the SSR summed over h = 0–4. The grid has 15 centres over the range of `inst_lag1` and lambda in `logspace(-0.5, 2, 12)`.
  - *Binary threshold:* the same equation with `1(inst_mean <= gamma)`, where gamma comes from a 40-point grid between the 10th and 90th percentiles of `inst_mean`.
- **Controls.**
  - *Baseline:* two lags of `gdp_growth` and `cpi_infl`.
  - *Extended:* adds one lag of geopolitical risk, energy imports, world GDP and the oil share.
  - *Further robustness sets:* a linear trend, or lags of the primary balance and the shock.
- **Inference.** Country block bootstrap: draw b resamples countries with replacement using `np.random.default_rng(42 + b)` and re-runs the grid search. Bands are 68/90/95% percentile intervals.

## Data

**Supplied inputs** (place them in `data/`):

| File | Content |
|---|---|
| `data/full-panel-data.db` | SQLite table `panel`: 33 countries, 1970–2026, 1,805 rows. This is the output of the upstream data-construction stage. |
| `data/bootstrapped-data/*.pkl` | Completed point estimates and bootstrap draws (joblib pickles), described below |
| `data/bootstrapped-data/panel_annual.parquet` | Processed copy of the panel. It is rebuilt from the database and used only as a consistency check. |

**Upstream sources (not included).**

- IMF: primary balance, expenditure, general government revenue, gross debt.
- World Bank WDI: GDP, CPI, tax revenue, GDP deflator, net energy imports, Brent price.
- V-Dem: institutional indicators.
- Känzig: oil supply news shock.
- Caldara and Iacoviello: geopolitical risk.
- Energy-mix data: oil share of primary energy.

The raw-data stage is documented in the original `data_prep.py`, but it cannot be run here because the raw files are not supplied.

**Sample.**

- The panel covers 33 countries.
- 31 of them have the institutional index. Mexico and Singapore have no V-Dem or fiscal data in the panel.
- 30 countries enter the regressions (Egypt has no primary balance data), over 1975–2024.
- Sample restrictions inherited from the upstream stage: Brazil is dropped after 2006, the United States after 2019, and Argentina and Hong Kong are excluded.

## Supplied estimation files

| File(s) | Specification | Used for |
|---|---|---|
| `uni_smooth_base_corrected.pkl` | Smooth transition, baseline controls, final grid (100 draws) | **Main result** |
| `baseline_binary_long_parallel.pkl` | Binary threshold, baseline controls (1,000 draws) | Baseline |
| `uni_smooth_ext_corrected.pkl` | Smooth transition, extended controls | Robustness |
| `robustness_extended_parallel.pkl`, `robustness_extended_v2_parallel.pkl` | Binary threshold, extended controls (40- and 50-point grids) | Robustness |
| `robustness_trend_parallel.pkl` | Binary threshold, extended controls and a linear trend | Robustness |
| `baseline_binary_pb_shock_controls.pkl`, `uni_smooth_pb_shock_controls.pkl` | Adding lags of the primary balance and the shock | Robustness |
| `exp_regime_parallel.pkl`, `exp_common_regime_parallel.pkl` | Energy-import exposure (regime-specific or common slope, 1 SD shock) at the baseline gamma | Robustness |
| `decomp_<outcome>_<base/ext>_parallel.pkl` | Binary LP at the baseline gamma for 10 fiscal outcomes | Decomposition |
| `st_boot_<sub-index>_<outcome>_parallel.pkl` | Smooth transition on each V-Dem sub-index (contemporaneous) for pb, expenditure, tax revenue and debt; evaluated at the 10th/90th percentiles | Sub-index analysis |
| `uni_smooth_base_parallel.pkl`, `uni_smooth_ext_parallel.pkl` | Earlier smooth-transition grid (optimum on the grid boundary c = -1) | Superseded, checked only |
| `baseline_binary_long.pkl` | Earlier sequential run on an older panel vintage | Not used |

The mapping is defined in `src/specifications.py`.

## Structure

```
├── README.md
├── requirements.txt
├── run_replication.py         # entry point
├── data/                      # supplied inputs (see above)
├── src/
│   ├── config.py              # paths, control sets, grids, seed
│   ├── data.py                # load the panel, construct estimation variables, sample
│   ├── institutions.py        # composite index, country means, correlations
│   ├── local_projections.py   # LP estimation, transition function, grid searches
│   ├── specifications.py      # each supplied file and the model behind it
│   ├── bootstrap_results.py   # load supplied results, bands, tests
│   ├── bootstrap.py           # optional re-run of the country block bootstrap
│   ├── tables.py
│   └── figures.py
└── outputs/
    ├── tables/
    └── figures/
```

## Running

```bash
pip install -r requirements.txt

python run_replication.py                          # everything, about 3 minutes
python run_replication.py --section data           # sample, descriptives, institutions
python run_replication.py --section baseline       # main smooth-transition and binary results
python run_replication.py --section robustness
python run_replication.py --section decomposition
```

The default run re-estimates every point estimate from the panel (including all grid searches) and takes the bootstrap draws from the supplied files. Re-running a bootstrap is optional and slow. The main smooth-transition specification needs roughly 8 CPU-seconds per draw.

```bash
python run_replication.py --section bootstrap --spec st_base --n-boot 100
```

This writes to `outputs/bootstrap_rerun/` and compares the new draws with the supplied ones.

## Outputs

**Tables** (`outputs/tables/`, CSV):

- `main_smooth_transition_parameters.csv`, `main_smooth_transition_irfs.csv`: c, lambda, transition width, and the regime IRFs with 68/90/95% bands.
- `binary_threshold_irfs.csv`, `binary_threshold_joint_test.csv`
- `robustness_summary_h4.csv`, `subindex_smooth_transition.csv`, `decomposition_h4.csv`
- `sample_by_country.csv`, `descriptive_statistics.csv`, `institutional_index_by_country.csv`, `institutional_correlations.csv`, `adf_unit_root.csv`
- `replication_check.csv`: recomputed versus supplied estimates for every file.
- `data_consistency_check.csv`: the rebuilt panel versus `panel_annual.parquet`.

**Figures** (`outputs/figures/`): main smooth-transition IRFs, transition function, response curves at h = 4 and 8, binary IRFs, robustness comparisons, exposure interactions, decomposition, sub-index IRFs, institutional index and oil shock.
