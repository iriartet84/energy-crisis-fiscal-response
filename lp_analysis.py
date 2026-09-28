import os
import warnings

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from joblib import Parallel, delayed

from data_prep import load_panel, base_dir, cache_dir, table_dir, figure_dir, inst_labels

# Extreme transition speeds in the grid search give near-collinear regressors
warnings.filterwarnings("ignore", message="The design matrix is rank-deficient")

# Set to True to rerun the bootstraps instead of loading the saved draws (slow)
RERUN_BOOTSTRAP = False
N_BOOT = 300
rerun_dir = os.path.join(base_dir, "outputs", "bootstrap_rerun")

HORIZON = 10

controls_base = ["gdp_growth", "cpi_infl"]
controls_ext = ["gdp_growth", "cpi_infl", "gpr", "energy_imports", "world_gdp_growth", "oil_share"]
main_vars = ["gdp_growth", "cpi_infl"]
main_vars_pb_shock = ["gdp_growth", "cpi_infl", "pb", "shock"]

# Smooth transition grids: final grid and the earlier grid (optimum on its boundary)
speeds = np.logspace(-0.5, 2, 12)
centers_old = np.linspace(-1.0, 1.0, 12)
speeds_old = [0.5, 1.0, 2.0, 3.0, 5.0]

LOW, HIGH = "#c0392b", "#2980b9"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

checks = []


def prepare_panel(panel, control_vars, n_lags_extra=0, lag_main=main_vars, outcome="pb", max_h=HORIZON):

    # Lags of the controls and cumulative changes y(t+h) - y(t)

    df = panel.copy().set_index(["country", "date"])
    lag_cols = []
    for v in control_vars:
        lags = 2 if v in lag_main else n_lags_extra
        for lag in range(1, lags + 1):
            col = f"{v}_lag{lag}"
            df[col] = df.groupby(level=0)[v].shift(lag)
            lag_cols.append(col)

    for h in range(max_h + 1):
        df[f"dh_{outcome}_h{h}"] = df.groupby(level=0)[outcome].shift(-h) - df[outcome]

    return df, lag_cols


def fit_fe(df, dep, reg_cols):

    # Country fixed effects through the within transformation

    df_h = df.dropna(subset=[dep] + reg_cols)
    if len(df_h) < 10:
        return None

    cols = [dep] + reg_cols
    dem = df_h[cols] - df_h[cols].groupby(level=0).transform("mean")
    return sm.OLS(dem[dep], sm.add_constant(dem[reg_cols])).fit()


def lp(df, shock_cols, lag_cols, outcome="pb", horizon=HORIZON):
    betas = {k: [] for k in shock_cols}
    for h in range(horizon + 1):
        mod = fit_fe(df, f"dh_{outcome}_h{h}", shock_cols + lag_cols)
        for k in shock_cols:
            betas[k].append(np.nan if mod is None else mod.params[k])
    return {k: np.array(v) for k, v in betas.items()}


def ssr(df, shock_cols, lag_cols, outcome="pb", max_h=4):

    # Grid search objective: SSR summed over h = 0, ..., 4

    total = 0.0
    for h in range(max_h + 1):
        mod = fit_fe(df, f"dh_{outcome}_h{h}", shock_cols + lag_cols)
        if mod is None:
            return np.inf
        total += mod.ssr
    return total


def logistic(z, center, speed):
    return 1 / (1 + np.exp(-speed * (z - center)))


def add_binary(df, gamma, shock="shock"):
    work = df.copy()
    low = (work["inst_mean"] <= gamma).astype(int)
    work["shock_low"] = work[shock] * low
    work["shock_high"] = work[shock] * (1 - low)
    return work


def add_smooth(df, center, speed, state="inst_lag1"):
    work = df.copy()
    F = logistic(work[state], center, speed)
    work["shock_F"] = work["shock"] * F
    work["shock_1mF"] = work["shock"] * (1 - F)
    return work


def add_exposure(df, gamma, common=False):

    # Standardised shock interacted with centred energy imports

    work = add_binary(df, gamma, shock="shock_std")
    low = (work["inst_mean"] <= gamma).astype(int)
    if common:
        work["shock_imports"] = work["shock_std"] * work["imports_centered"]
    else:
        work["shock_imports_low"] = work["shock_std"] * work["imports_centered"] * low
        work["shock_imports_high"] = work["shock_std"] * work["imports_centered"] * (1 - low)
    return work


def lp_binary(df, lag_cols, candidates=None, gamma=None, outcome="pb"):

    # Grid search for the threshold unless it is fixed

    if gamma is None:
        best_ssr = np.inf
        for g in candidates:
            s = ssr(add_binary(df, g), ["shock_low", "shock_high"], lag_cols, outcome)
            if s < best_ssr:
                best_ssr, gamma = s, g

    betas = lp(add_binary(df, gamma), ["shock_low", "shock_high"], lag_cols, outcome)
    return {"best_gamma": gamma, "b_low": betas["shock_low"], "b_high": betas["shock_high"]}


def lp_smooth(df, lag_cols, centers, speeds, state="inst_lag1", outcome="pb"):

    # Grid search over the center and speed of the transition

    best_ssr, best_ci, best_si = np.inf, None, None
    for c in centers:
        for s in speeds:
            value = ssr(add_smooth(df, c, s, state), ["shock_F", "shock_1mF"], lag_cols, outcome)
            if value < best_ssr:
                best_ssr, best_ci, best_si = value, c, s

    betas = lp(add_smooth(df, best_ci, best_si, state), ["shock_F", "shock_1mF"], lag_cols, outcome)
    return {"best_ci": best_ci, "best_si": best_si, "ssr": best_ssr,
            "bF": betas["shock_F"], "b1mF": betas["shock_1mF"]}


def lp_exposure(df, lag_cols, gamma, common=False):
    if common:
        cols = ["shock_low", "shock_high", "shock_imports"]
    else:
        cols = ["shock_low", "shock_high", "shock_imports_low", "shock_imports_high"]
    return lp(add_exposure(df, gamma, common), cols, lag_cols)


def single_bootstrap(seed, country_cache, estimate):

    # Resample whole countries with replacement and re-estimate

    rng = np.random.default_rng(seed)
    countries = list(country_cache.keys())
    draw = rng.choice(countries, size=len(countries), replace=True)
    df_boot = pd.concat([country_cache[c] for c in draw])
    return estimate(df_boot)


def run_bootstrap(prepared, estimate, n_boot=N_BOOT):
    country_cache = {c: grp for c, grp in prepared.groupby(level=0)}
    return Parallel(n_jobs=-1, backend="loky", verbose=5)(
        delayed(single_bootstrap)(42 + b, country_cache, estimate) for b in range(n_boot)
    )


def cached_bootstrap(name, compute_func):

    # Load the saved results unless the bootstrap is rerun

    if not RERUN_BOOTSTRAP:
        return joblib.load(os.path.join(cache_dir, name))

    print(f"   Running {name} ...")
    result = compute_func()
    os.makedirs(rerun_dir, exist_ok=True)
    joblib.dump(result, os.path.join(rerun_dir, name))
    return result


def bands(arr, level=68):
    lo, hi = {68: (16, 84), 90: (5, 95), 95: (2.5, 97.5)}[level]
    return np.percentile(arr, lo, axis=0), np.percentile(arr, hi, axis=0)


def irf_table(point, boot, name):
    table = pd.DataFrame({"horizon": np.arange(len(point)), name: point})
    for level in [68, 90, 95]:
        table[f"{name}_lo{level}"], table[f"{name}_hi{level}"] = bands(boot, level)
    return table


def add_check(name, supplied, recomputed, keys, param=""):

    # Compare the re-estimated point estimates with the saved file

    diff = max(np.nanmax(np.abs(np.asarray(supplied[k]) - np.asarray(recomputed[k]))) for k in keys)
    checks.append({"file": name, "parameters": param, "max_abs_diff": diff})


def plot_irf(ax, irf, boot, color, label, marker="o", linestyle="-", levels=(95, 68)):
    h = np.arange(len(irf))
    alpha = {95: 0.12, 68: 0.28}
    for level in levels:
        lo, hi = bands(boot, level)
        ax.fill_between(h, lo, hi, color=color, alpha=alpha[level], linewidth=0)
    ax.plot(h, irf, marker=marker, linestyle=linestyle, color=color, linewidth=2, markersize=4, label=label)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Years after shock")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))


def plot_regimes(low, boot_low, high, boot_high, title, titles, filename):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    plot_irf(axes[0], low, boot_low, LOW, "Point estimate")
    plot_irf(axes[1], high, boot_high, HIGH, "Point estimate", marker="s")
    axes[0].set_title(titles[0])
    axes[1].set_title(titles[1])
    axes[0].set_ylabel("Cum. PB response (% GDP)")
    axes[0].legend(frameon=False)
    fig.suptitle(title + "\n(shaded: 68% and 95% bootstrap bands)")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, filename), dpi=200)
    plt.close()


def plot_comparison(models, title, filename):

    # Several specifications in each regime, 68% bands only

    colors = ["#e67e22", "#27ae60", "#8e44ad"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for i, (label, low, boot_low, high, boot_high) in enumerate(models):
        style = "-" if i == 0 else "--"
        plot_irf(axes[0], low, boot_low, LOW if i == 0 else colors[i - 1], label, linestyle=style, levels=(68,))
        plot_irf(axes[1], high, boot_high, HIGH if i == 0 else colors[i - 1], label, marker="s",
                 linestyle=style, levels=(68,))

    axes[0].set_title("Low institutions")
    axes[1].set_title("High institutions")
    axes[0].set_ylabel("Cum. PB response (% GDP)")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.suptitle(title + "\n(shaded: 68% bootstrap bands)")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, filename), dpi=200)
    plt.close()


def compute_binary(prepared, lag_cols, candidates, outcome="pb"):
    result = lp_binary(prepared, lag_cols, candidates=candidates, outcome=outcome)
    draws = run_bootstrap(prepared, lambda d: lp_binary(d, lag_cols, candidates=candidates, outcome=outcome))
    result["irf_low_boot"] = np.vstack([r["b_low"] for r in draws])
    result["irf_high_boot"] = np.vstack([r["b_high"] for r in draws])
    result["gamma_boot"] = [r["best_gamma"] for r in draws]
    return result


def compute_fixed(prepared, lag_cols, gamma, outcome):
    result = lp_binary(prepared, lag_cols, gamma=gamma, outcome=outcome)
    draws = run_bootstrap(prepared, lambda d: lp_binary(d, lag_cols, gamma=gamma, outcome=outcome))
    result["irf_low_boot"] = np.vstack([r["b_low"] for r in draws])
    result["irf_high_boot"] = np.vstack([r["b_high"] for r in draws])
    return result


def compute_smooth(prepared, lag_cols, centers, speeds):
    result = lp_smooth(prepared, lag_cols, centers, speeds)
    draws = run_bootstrap(prepared, lambda d: lp_smooth(d, lag_cols, centers, speeds))
    result["b_F_boot"] = np.vstack([r["bF"] for r in draws])
    result["b_1mF_boot"] = np.vstack([r["b1mF"] for r in draws])
    result["ci_boot"] = np.array([r["best_ci"] for r in draws])
    result["si_boot"] = np.array([r["best_si"] for r in draws])
    return result


def compute_exposure(prepared, lag_cols, gamma, common):
    betas = lp_exposure(prepared, lag_cols, gamma, common)
    draws = run_bootstrap(prepared, lambda d: lp_exposure(d, lag_cols, gamma, common))
    result = {"betas": betas,
              "boot_low": np.vstack([r["shock_low"] for r in draws]),
              "boot_high": np.vstack([r["shock_high"] for r in draws])}
    if common:
        result["boot_imports"] = np.vstack([r["shock_imports"] for r in draws])
    else:
        result["boot_imp_low"] = np.vstack([r["shock_imports_low"] for r in draws])
        result["boot_imp_high"] = np.vstack([r["shock_imports_high"] for r in draws])
    return result


def smooth_at(result, z):

    # Smooth-transition IRF at a given level of lagged institutions

    F = logistic(z, result["best_ci"], result["best_si"])
    irf = result["bF"] * F + result["b1mF"] * (1 - F)
    boot = result["b_F_boot"] * F + result["b_1mF_boot"] * (1 - F)
    return irf, boot


def regime_means(panel, center):
    x = panel["inst_lag1"].dropna()
    return x[x <= center].mean(), x[x > center].mean()


if __name__ == "__main__":
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    # Load the panel

    panel = load_panel()

    inst_vals = panel["inst_mean"].dropna()
    candidates = np.linspace(np.percentile(inst_vals, 10), np.percentile(inst_vals, 90), 40)
    centers = np.linspace(panel["inst_lag1"].min(), panel["inst_lag1"].max(), 15)

    prepared_base, lag_cols_base = prepare_panel(panel, controls_base)
    prepared_ext, lag_cols_ext = prepare_panel(panel, controls_ext, n_lags_extra=1)

    # 1. Binary threshold LP, baseline controls

    print("1. Binary threshold LP")

    base = cached_bootstrap("baseline_binary_long_parallel.pkl",
                            lambda: compute_binary(prepared_base, lag_cols_base, candidates))
    base_re = lp_binary(prepared_base, lag_cols_base, candidates=candidates)
    add_check("baseline_binary_long_parallel.pkl", base, base_re, ["b_low", "b_high"],
              f"gamma = {base['best_gamma']:.3f} (re-estimated {base_re['best_gamma']:.3f})")

    best_gamma = base["best_gamma"]
    print(f"   gamma = {best_gamma:.3f}, h=4: low = {base['b_low'][4]:.3f}, high = {base['b_high'][4]:.3f}")

    # Pointwise p-values and joint max-t test for the difference between regimes

    diff_boot = base["irf_high_boot"] - base["irf_low_boot"]
    diff_point = base["b_high"] - base["b_low"]

    table = irf_table(base["b_low"], base["irf_low_boot"], "low").merge(
        irf_table(base["b_high"], base["irf_high_boot"], "high"), on="horizon")
    table["difference"] = diff_point
    table["pvalue_difference"] = np.mean(np.abs(diff_boot) >= np.abs(diff_point), axis=0)
    table.to_csv(os.path.join(table_dir, "binary_threshold_irfs.csv"), index=False)

    tests = []
    for label, hs in [("h = 0-4", list(range(5))), ("h = 0-10", list(range(11)))]:
        se = np.std(diff_boot[:, hs], axis=0, ddof=1)
        se[se < 1e-12] = 1e-12
        t_obs = np.max(np.abs(diff_point[hs] / se))
        t_boot = np.max(np.abs(diff_boot[:, hs] / se), axis=1)
        tests.append({"horizons": label, "max_t": t_obs, "pvalue": np.mean(t_boot >= t_obs)})
    pd.DataFrame(tests).to_csv(os.path.join(table_dir, "binary_threshold_joint_test.csv"), index=False)

    plot_regimes(base["b_low"], base["irf_low_boot"], base["b_high"], base["irf_high_boot"],
                 "Binary threshold LP (baseline controls)",
                 [f"Low institutions (inst mean <= {best_gamma:.2f})",
                  f"High institutions (inst mean > {best_gamma:.2f})"],
                 "binary_threshold_irfs.png")

    # 1. A) Robustness: extended controls, 50-point grid, time trend, PB and shock lags

    print("1. A) Robustness of the binary LP")

    candidates_50 = np.linspace(np.percentile(inst_vals, 10), np.percentile(inst_vals, 90), 50)
    prepared_trend, lag_cols_trend = prepare_panel(panel, controls_ext + ["trend"], n_lags_extra=1)
    prepared_pbs, lag_cols_pbs = prepare_panel(panel, main_vars_pb_shock, lag_main=main_vars_pb_shock)

    binary_specs = {
        "robustness_extended_parallel.pkl": (prepared_ext, lag_cols_ext, candidates),
        "robustness_extended_v2_parallel.pkl": (prepared_ext, lag_cols_ext, candidates_50),
        "robustness_trend_parallel.pkl": (prepared_trend, lag_cols_trend, candidates),
        "baseline_binary_pb_shock_controls.pkl": (prepared_pbs, lag_cols_pbs, candidates),
    }

    robust = {}
    for name, (df, cols, cand) in binary_specs.items():
        robust[name] = cached_bootstrap(name, lambda: compute_binary(df, cols, cand))
        re = lp_binary(df, cols, candidates=cand)
        add_check(name, robust[name], re, ["b_low", "b_high"], f"gamma = {robust[name]['best_gamma']:.3f}")

    def binary_entry(label, res):
        return label, res["b_low"], res["irf_low_boot"], res["b_high"], res["irf_high_boot"]

    plot_comparison([binary_entry("Baseline controls", base),
                     binary_entry("Extended controls", robust["robustness_extended_parallel.pkl"]),
                     binary_entry("Extended controls + trend", robust["robustness_trend_parallel.pkl"])],
                    "Binary threshold LP: alternative control sets", "robustness_binary_controls.png")

    pbs = robust["baseline_binary_pb_shock_controls.pkl"]
    plot_comparison([binary_entry("Baseline controls", base),
                     binary_entry(f"Plus PB and shock lags (gamma = {pbs['best_gamma']:.2f})", pbs)],
                    "Binary threshold LP: adding lags of the primary balance and the shock",
                    "robustness_binary_pb_shock_lags.png")

    # 1. B) Energy-import exposure at the baseline threshold

    print("1. B) Energy-import exposure")

    import_sd = panel["energy_imports"].std()
    exp_regime = cached_bootstrap("exp_regime_parallel.pkl",
                                  lambda: compute_exposure(prepared_base, lag_cols_base, best_gamma, False))
    exp_common = cached_bootstrap("exp_common_regime_parallel.pkl",
                                  lambda: compute_exposure(prepared_base, lag_cols_base, best_gamma, True))

    for name, res, common in [("exp_regime_parallel.pkl", exp_regime, False),
                              ("exp_common_regime_parallel.pkl", exp_common, True)]:
        re = lp_exposure(prepared_base, lag_cols_base, best_gamma, common)
        add_check(name, res["betas"], re, list(re.keys()), "gamma fixed at baseline")

    b = exp_regime["betas"]
    plot_comparison([
        ("Average imports", b["shock_low"], exp_regime["boot_low"], b["shock_high"], exp_regime["boot_high"]),
        ("+1 SD imports",
         b["shock_low"] + b["shock_imports_low"] * import_sd,
         exp_regime["boot_low"] + exp_regime["boot_imp_low"] * import_sd,
         b["shock_high"] + b["shock_imports_high"] * import_sd,
         exp_regime["boot_high"] + exp_regime["boot_imp_high"] * import_sd),
    ], "Energy-import exposure: regime-specific import slopes (1 SD oil shock)",
        "robustness_exposure_regime.png")

    b = exp_common["betas"]
    plot_comparison([
        ("Average imports", b["shock_low"], exp_common["boot_low"], b["shock_high"], exp_common["boot_high"]),
        ("+1 SD imports",
         b["shock_low"] + b["shock_imports"] * import_sd,
         exp_common["boot_low"] + exp_common["boot_imports"] * import_sd,
         b["shock_high"] + b["shock_imports"] * import_sd,
         exp_common["boot_high"] + exp_common["boot_imports"] * import_sd),
    ], "Energy-import exposure: common import slope (1 SD oil shock)",
        "robustness_exposure_common.png")

    # 2. Smooth-transition LP, baseline controls (main specification)

    print("2. Smooth-transition LP")

    st = cached_bootstrap("uni_smooth_base_corrected.pkl",
                          lambda: compute_smooth(prepared_base, lag_cols_base, centers, speeds))
    st_re = lp_smooth(prepared_base, lag_cols_base, centers, speeds)
    add_check("uni_smooth_base_corrected.pkl", st, st_re, ["bF", "b1mF"],
              f"c = {st['best_ci']:.3f}, lambda = {st['best_si']:.2f}")

    best_ci, best_si = st["best_ci"], st["best_si"]
    width = np.log(9) / best_si
    print(f"   c = {best_ci:.3f}, lambda = {best_si:.2f}, width = {width:.3f}")
    print(f"   h=4: low = {st['b1mF'][4]:.3f}, high = {st['bF'][4]:.3f} "
          f"(re-estimated {st_re['b1mF'][4]:.3f}, {st_re['bF'][4]:.3f})")

    # IRFs at the mean lagged institutions on each side of the center

    mean_low, mean_high = regime_means(panel, best_ci)
    irf_low, boot_low = smooth_at(st, mean_low)
    irf_high, boot_high = smooth_at(st, mean_high)

    table = irf_table(irf_low, boot_low, "low").merge(irf_table(irf_high, boot_high, "high"), on="horizon")
    table["b1mF_recomputed"] = st_re["b1mF"]
    table["bF_recomputed"] = st_re["bF"]
    table.to_csv(os.path.join(table_dir, "main_smooth_transition_irfs.csv"), index=False)

    params = pd.DataFrame([
        ("center c", best_ci, st_re["best_ci"]),
        ("speed lambda", best_si, st_re["best_si"]),
        ("transition width ln(9)/lambda", width, np.log(9) / st_re["best_si"]),
        ("SSR over h = 0-4", np.nan, st_re["ssr"]),
        ("mean lagged inst, low side", mean_low, np.nan),
        ("mean lagged inst, high side", mean_high, np.nan),
        ("low-state coefficient b1mF, h = 4", st["b1mF"][4], st_re["b1mF"][4]),
        ("high-state coefficient bF, h = 4", st["bF"][4], st_re["bF"][4]),
        ("bootstrap median c", np.median(st["ci_boot"]), np.nan),
        ("bootstrap median lambda", np.median(st["si_boot"]), np.nan),
        ("bootstrap draws", len(st["ci_boot"]), np.nan),
    ], columns=["statistic", "supplied", "recomputed"])
    params.to_csv(os.path.join(table_dir, "main_smooth_transition_parameters.csv"), index=False)

    plot_regimes(irf_low, boot_low, irf_high, boot_high,
                 "Smooth-transition LP: primary balance response to an oil supply shock",
                 [f"Low institutions (mean lagged inst = {mean_low:.2f})",
                  f"High institutions (mean lagged inst = {mean_high:.2f})"],
                 "main_smooth_transition_irfs.png")

    # Transition function

    grid = np.linspace(panel["inst_lag1"].min(), panel["inst_lag1"].max(), 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, logistic(grid, best_ci, best_si), color="black", linewidth=2)
    ax.plot(panel["inst_lag1"].dropna(), np.full(panel["inst_lag1"].notna().sum(), -0.04), "|",
            color="gray", alpha=0.3)
    ax.axvline(best_ci, color="gray", linestyle=":")
    ax.set_xlabel("Lagged institutional quality")
    ax.set_ylabel("F(inst)")
    ax.set_title(f"Logistic transition function (c = {best_ci:.3f}, lambda = {best_si:.1f})")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "transition_function.png"), dpi=200)
    plt.close()

    # Response across institutional quality at h = 4 and h = 8, using each draw's own (c, lambda)

    inst_grid = np.linspace(-1.5, 1.5, 100)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, h in zip(axes, [4, 8]):
        F = logistic(inst_grid[None, :], st["ci_boot"][:, None], st["si_boot"][:, None])
        mult = st["b_F_boot"][:, [h]] * F + st["b_1mF_boot"][:, [h]] * (1 - F)
        lo, hi = bands(mult)
        ax.fill_between(inst_grid, lo, hi, color="steelblue", alpha=0.3, linewidth=0, label="68% band")
        ax.plot(inst_grid, np.median(mult, axis=0), color="navy", linewidth=2, label="Bootstrap median")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(best_ci, color="gray", linestyle=":", label=f"ST center c = {best_ci:.2f}")
        ax.axvline(best_gamma, color="black", linestyle="--", linewidth=1,
                   label=f"Binary threshold = {best_gamma:.2f}")
        ax.set_xlabel("Lagged institutional quality (standardised)")
        ax.set_title(f"Horizon h = {h}")
    axes[0].set_ylabel("Cum. PB response (% GDP)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Smooth-transition fiscal response across institutional quality")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "smooth_transition_response_curve.png"), dpi=200)
    plt.close()

    # 2. A) Robustness: extended controls, PB and shock lags, earlier grid

    print("2. A) Robustness of the smooth-transition LP")

    prepared_st_pbs, lag_cols_st_pbs = prepare_panel(panel, controls_ext + ["pb", "shock"], n_lags_extra=1,
                                                     lag_main=main_vars_pb_shock)

    smooth_specs = {
        "uni_smooth_ext_corrected.pkl": (prepared_ext, lag_cols_ext, centers, speeds),
        "uni_smooth_pb_shock_controls.pkl": (prepared_st_pbs, lag_cols_st_pbs, centers_old, speeds_old),
        "uni_smooth_base_parallel.pkl": (prepared_base, lag_cols_base, centers_old, speeds_old),
        "uni_smooth_ext_parallel.pkl": (prepared_ext, lag_cols_ext, centers_old, speeds_old),
    }

    smooth = {"uni_smooth_base_corrected.pkl": st}
    for name, (df, cols, cs, ss) in smooth_specs.items():
        smooth[name] = cached_bootstrap(name, lambda: compute_smooth(df, cols, cs, ss))
        re = lp_smooth(df, cols, cs, ss)
        add_check(name, smooth[name], re, ["bF", "b1mF"],
                  f"c = {smooth[name]['best_ci']:.3f}, lambda = {smooth[name]['best_si']:.2f}")

    models = []
    for name, label in [("uni_smooth_base_corrected.pkl", "Baseline controls"),
                        ("uni_smooth_ext_corrected.pkl", "Extended controls"),
                        ("uni_smooth_pb_shock_controls.pkl", "Extended + PB and shock lags")]:
        res = smooth[name]
        z_low, z_high = regime_means(panel, res["best_ci"])
        low, blow = smooth_at(res, z_low)
        high, bhigh = smooth_at(res, z_high)
        models.append((f"{label} (c = {res['best_ci']:.2f})", low, blow, high, bhigh))

    plot_comparison(models, "Smooth-transition LP: alternative control sets", "robustness_smooth_transition.png")

    # Summary of all specifications at h = 4

    summary = {
        "uni_smooth_base_corrected.pkl": st,
        "uni_smooth_ext_corrected.pkl": smooth["uni_smooth_ext_corrected.pkl"],
        "uni_smooth_pb_shock_controls.pkl": smooth["uni_smooth_pb_shock_controls.pkl"],
        "baseline_binary_long_parallel.pkl": base,
        **robust,
    }

    rows = []
    for name, res in summary.items():
        if "bF" in res:
            low, high, blow, bhigh = res["b1mF"], res["bF"], res["b_1mF_boot"], res["b_F_boot"]
            param = f"c = {res['best_ci']:.3f}, lambda = {res['best_si']:.2f}"
        else:
            low, high, blow, bhigh = res["b_low"], res["b_high"], res["irf_low_boot"], res["irf_high_boot"]
            param = f"gamma = {res['best_gamma']:.3f}"
        rows.append({"file": name, "parameters": param,
                     "low_h4": low[4], "low_h4_lo68": bands(blow[:, 4])[0], "low_h4_hi68": bands(blow[:, 4])[1],
                     "high_h4": high[4], "high_h4_lo68": bands(bhigh[:, 4])[0],
                     "high_h4_hi68": bands(bhigh[:, 4])[1], "n_boot": blow.shape[0]})

    for name, res in [("exp_regime_parallel.pkl", exp_regime), ("exp_common_regime_parallel.pkl", exp_common)]:
        rows.append({"file": name, "parameters": f"gamma = {best_gamma:.3f} (fixed), 1 SD shock",
                     "low_h4": res["betas"]["shock_low"][4],
                     "low_h4_lo68": bands(res["boot_low"][:, 4])[0], "low_h4_hi68": bands(res["boot_low"][:, 4])[1],
                     "high_h4": res["betas"]["shock_high"][4],
                     "high_h4_lo68": bands(res["boot_high"][:, 4])[0],
                     "high_h4_hi68": bands(res["boot_high"][:, 4])[1], "n_boot": res["boot_low"].shape[0]})

    pd.DataFrame(rows).to_csv(os.path.join(table_dir, "robustness_summary_h4.csv"), index=False)

    # 3. Fiscal decomposition at the baseline threshold

    print("3. Fiscal decomposition")

    decomp_labels = {
        "pb": "Primary balance (% GDP)",
        "gov_revenue": "General government revenue (% GDP)",
        "tax_rev_gdp": "Tax revenue (% GDP)",
        "expenditure": "Expenditure (% GDP)",
        "debt": "Gross debt (% GDP)",
        "ihs_pb": "IHS real primary balance",
        "log_real_gov_rev": "Log real government revenue",
        "log_real_tax": "Log real tax revenue",
        "log_expenditure": "Log real expenditure",
        "log_debt": "Log real debt",
    }

    decomp = {}
    rows = []
    for outcome, label in decomp_labels.items():
        for spec, ctrl, n_extra in [("base", controls_base, 0), ("ext", controls_ext, 1)]:
            name = f"decomp_{outcome}_{spec}_parallel.pkl"
            df, cols = prepare_panel(panel, ctrl, n_lags_extra=n_extra, outcome=outcome)
            res = cached_bootstrap(name, lambda: compute_fixed(df, cols, best_gamma, outcome))
            add_check(name, res, lp_binary(df, cols, gamma=best_gamma, outcome=outcome), ["b_low", "b_high"],
                      "gamma fixed at baseline")
            decomp[(outcome, spec)] = res

            d_boot = res["irf_high_boot"] - res["irf_low_boot"]
            d_point = res["b_high"] - res["b_low"]
            rows.append({"outcome": outcome, "controls": spec, "label": label,
                         "low_h4": res["b_low"][4], "low_h4_lo68": bands(res["irf_low_boot"][:, 4])[0],
                         "low_h4_hi68": bands(res["irf_low_boot"][:, 4])[1],
                         "high_h4": res["b_high"][4], "high_h4_lo68": bands(res["irf_high_boot"][:, 4])[0],
                         "high_h4_hi68": bands(res["irf_high_boot"][:, 4])[1],
                         "difference_h4": d_point[4],
                         "pvalue_difference_h4": np.mean(np.abs(d_boot[:, 4]) >= np.abs(d_point[4]))})

    pd.DataFrame(rows).to_csv(os.path.join(table_dir, "decomposition_h4.csv"), index=False)

    groups = {"pct_gdp": ["pb", "gov_revenue", "tax_rev_gdp", "expenditure", "debt"],
              "real": ["ihs_pb", "log_real_gov_rev", "log_real_tax", "log_expenditure", "log_debt"]}

    for spec in ["base", "ext"]:
        for group, outcomes in groups.items():
            fig, axes = plt.subplots(len(outcomes), 2, figsize=(12, 3.2 * len(outcomes)))
            for row, outcome in enumerate(outcomes):
                res = decomp[(outcome, spec)]
                plot_irf(axes[row, 0], res["b_low"], res["irf_low_boot"], LOW, "Low")
                plot_irf(axes[row, 1], res["b_high"], res["irf_high_boot"], HIGH, "High", marker="s")
                axes[row, 0].set_ylabel(decomp_labels[outcome], fontsize=9)
            axes[0, 0].set_title("Low institutions")
            axes[0, 1].set_title("High institutions")
            fig.suptitle(f"Fiscal decomposition at gamma = {best_gamma:.2f} ({spec} controls)\n"
                         "(shaded: 68% and 95% bootstrap bands)")
            plt.tight_layout()
            plt.savefig(os.path.join(figure_dir, f"decomposition_{group}_{spec}.png"), dpi=200)
            plt.close()

    # 4. Smooth transition by institutional sub-index
    # The bootstrap for these files is not in this script, so the saved draws are always used

    print("4. Sub-index smooth transitions")

    sub_indices = ["control_of_corruption", "rule_of_law", "property_rights",
                   "transparent_laws", "bureaucratic_quality", "democracy"]
    speeds_sub = np.logspace(-0.5, 2, 8)

    rows = []
    for outcome in ["pb", "expenditure", "tax_rev_gdp", "debt"]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), sharey=True)
        df, cols = prepare_panel(panel, controls_base, outcome=outcome)

        for ax, var in zip(axes.flat, sub_indices):
            name = f"st_boot_{var}_{outcome}_parallel.pkl"
            res = joblib.load(os.path.join(cache_dir, name))

            # b_high is the coefficient in the high (F = 1) state
            vals = panel[var].dropna()
            centers_sub = np.linspace(np.percentile(vals, 5), np.percentile(vals, 95), 20)
            re = lp_smooth(df, cols, centers_sub, speeds_sub, state=var, outcome=outcome)
            add_check(name, {"bF": res["b_high"], "b1mF": res["b_low"]}, re, ["bF", "b1mF"],
                      f"c = {res['c_opt']:.3f}, lambda = {res['gamma_opt']:.2f}")

            on_edge = (np.isclose(res["c_opt"], centers_sub[[0, -1]]).any()
                       or np.isclose(res["gamma_opt"], speeds_sub[[0, -1]]).any())
            lo_l, lo_h = bands(res["boot_irf_low"][:, 4])
            hi_l, hi_h = bands(res["boot_irf_high"][:, 4])
            rows.append({"sub_index": var, "outcome": outcome, "c": res["c_opt"], "lambda": res["gamma_opt"],
                         "ssr": res["SSR"], "p10": res["p10"], "p90": res["p90"],
                         "at_p10_h4": res["irf_low"][4], "at_p10_h4_lo68": lo_l, "at_p10_h4_hi68": lo_h,
                         "at_p90_h4": res["irf_high"][4], "at_p90_h4_lo68": hi_l, "at_p90_h4_hi68": hi_h,
                         "on_grid_boundary": bool(on_edge), "n_boot": res["boot_irf_low"].shape[0]})

            plot_irf(ax, res["irf_low"], res["boot_irf_low"], LOW, "10th percentile", levels=(68,))
            plot_irf(ax, res["irf_high"], res["boot_irf_high"], HIGH, "90th percentile", marker="s",
                     levels=(68,))
            ax.set_title(f"{inst_labels[var]} (c = {res['c_opt']:.2f}, lambda = {res['gamma_opt']:.1f})",
                         fontsize=10)

        axes[0, 0].set_ylabel(f"Cum. response of {outcome}")
        axes[0, 0].legend(frameon=False)
        fig.suptitle(f"Smooth transition by institutional sub-index, outcome: {outcome}\n"
                     "(shaded: 68% bootstrap bands)")
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, f"subindex_smooth_transition_{outcome}.png"), dpi=200)
        plt.close()

    pd.DataFrame(rows).to_csv(os.path.join(table_dir, "subindex_smooth_transition.csv"), index=False)

    # Save the comparison with the saved estimates

    check_table = pd.DataFrame(checks)
    check_table.to_csv(os.path.join(table_dir, "replication_check.csv"), index=False)
    print(f"Re-estimated {len(check_table)} saved results, "
          f"largest difference = {check_table['max_abs_diff'].max():.1e}")
