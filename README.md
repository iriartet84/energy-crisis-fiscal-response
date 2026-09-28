# Energy Shocks and Fiscal Policy

## Overview

This Master's thesis examines whether **institutional quality conditions the fiscal response to exogenous oil supply shocks**.

The project combines international macroeconomic data with measures of political and institutional quality to study whether countries with stronger institutions respond differently to energy-driven economic shocks. The analysis uses state-dependent local projections to estimate how fiscal balances evolve following oil supply disruptions across different institutional environments.

The study covers **29 countries over 1974–2024**, combining oil supply shocks, fiscal outcomes, institutional indicators, and macroeconomic controls.

## Research Question

> **Does institutional quality shape the fiscal response to exogenous oil supply shocks?**

The central hypothesis is that institutional characteristics can affect how governments absorb and respond to energy shocks through fiscal policy.

Rather than estimating a single average response, the analysis allows the fiscal effect of an oil shock to vary continuously with institutional quality.

## Methodology

The empirical strategy combines:

* **State-dependent local projections**
* **Oil supply shocks** as the exogenous energy shock
* **Institutional-quality measures** constructed from multiple governance indicators
* **Panel data covering 29 countries from 1974–2024**
* Country and time controls
* Bootstrap inference

The baseline fiscal outcome is the **primary balance**, transformed using the inverse hyperbolic sine transformation to accommodate observations around zero.

The institutional index combines measures including:

* Control of corruption
* Property rights
* Transparent laws
* Democracy
* Rule of law

The institutional dimensions are combined into a composite measure to capture broader differences in institutional quality across countries and over time.

## State-Dependent Local Projections

The main empirical framework allows the response to an oil supply shock to vary with institutional quality.

For horizon \(h\), the fiscal response is estimated as a function of the oil shock and a smooth transition between lower- and higher-institutional-quality states.

This approach avoids imposing a discrete classification of countries into "strong" and "weak" institutions and instead estimates how the response changes continuously across the institutional distribution.

The transition function is logistic, allowing the marginal effect of the oil shock to change around an estimated institutional threshold.

## Data

The analysis combines several international datasets:

| Dataset        | Main variables                              |
| -------------- | ------------------------------------------- |
| Oil shock data | Exogenous oil supply shocks                 |
| IMF            | Fiscal balances and macroeconomic variables |
| World Bank     | Macroeconomic and structural controls       |
| V-Dem          | Institutional and political indicators      |

The final panel contains **29 countries covering 1974–2024**, subject to data availability.

## Key Results

The estimated transition function indicates meaningful state dependence in the fiscal response to oil supply shocks.

The baseline specification estimates a transition location around **−0.94** on the institutional index, with a relatively sharp transition between lower- and higher-institutional-quality states.

The estimated low- and high-institution responses differ substantially, with the fiscal response changing from approximately **−0.19** in the lower-institution state to **+0.16** in the higher-institution state at the relevant horizon.

The estimated threshold is subject to substantial uncertainty, however, with a bootstrap confidence interval approximately spanning **−1.85 to 1.33**. This uncertainty is important when interpreting the exact location of the institutional transition.

The broader result is therefore about **state dependence in fiscal responses**, rather than a precisely estimated universal institutional threshold.

## Empirical Workflow

The research pipeline follows:

```text
Oil Supply Shocks
        ↓
Country-Year Panel
        ↓
Institutional Quality Index
        ↓
Macro & Fiscal Controls
        ↓
State-Dependent Local Projections
        ↓
Bootstrap Inference
        ↓
Fiscal Response Across Institutional States
```
