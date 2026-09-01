# 🌍 Predicting Economic Development in Africa Using World Development Indicators

**Data Science Capstone Project — AnalystLab Africa Data Science Internship**

## Problem Statement

Can we predict a country's level of economic development (GDP per capita) using health,
education, investment, and infrastructure indicators, for African economies?

An earlier attempt to predict **year-over-year GDP growth (%)** instead of the GDP per capita
*level* produced very weak models (R² < 0.06), since annual growth is dominated by short-term
shocks that slow-moving development indicators cannot capture — this is documented in the
notebook for transparency. Predicting the GDP per capita **level** is a better-posed, still
highly meaningful question, since it directly measures how strongly health, education, and
infrastructure development track with economic development across the continent.

## Dataset

- **Source:** World Bank [World Development Indicators (WDI)](https://datatopics.worldbank.org/world-development-indicators/) — official bulk download (`WDI_CSV.zip` → `WDICSV.csv`, plus `WDICountry.csv` for region metadata)
- **Coverage:** 54 African countries, 2000–2020 (1,097 country-year rows after cleaning)
- **Indicators used:** GDP per capita, GDP growth, life expectancy, under-5 mortality, secondary
  school enrollment, government education expenditure, gross capital formation, FDI net inflows,
  electricity access, population growth, inflation, unemployment

> **Note on reproducing this notebook:** `WDICSV.csv` (~198MB) is the World Bank's official bulk
> file and is **not included in this repository** (it exceeds GitHub's recommended file size and
> is trivial to re-download). To re-run the notebook from scratch:
> 1. Go to https://datatopics.worldbank.org/world-development-indicators/
> 2. Access Data → Bulk Downloads → CSV Download → download `WDI_CSV.zip`
> 3. Extract it and place `WDICSV.csv` (and `WDICountry.csv`) in the `notebook/` folder
> 4. Run the notebook — it filters, reshapes, and cleans this file itself (Section 3)
>
> The already-cleaned, filtered dataset (`africa_wdi_clean.csv`, ~200KB) **is** included, so the
> modeling, evaluation, and deployment sections can be explored without re-downloading the full
> bulk file.

## Methodology

1. **Data Collection** — 12 WDI indicator series merged into a single country-year panel,
   filtered to Africa's 54 recognized countries.
2. **Data Cleaning** — missing target rows dropped; predictors imputed via country-level
   interpolation → country median → global median (in that priority order); extreme values in
   FDI, inflation, and GDP per capita winsorized at the 1st/99th percentile rather than dropped.
3. **EDA** — correlation heatmap, distribution analysis, country trend lines, scatter
   relationships between health/infrastructure indicators and GDP per capita.
4. **Modeling** — Linear Regression, Random Forest, and Gradient Boosting compared; Random
   Forest tuned via GridSearchCV with 5-fold cross-validation.
5. **Evaluation** — MAE, RMSE, R² on a held-out test set, cross-validated for robustness.
6. **Deployment** — FastAPI REST API and Streamlit web app, both backed by the same trained model.

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.423 | 0.558 | 0.735 |
| Random Forest (default) | 0.191 | 0.259 | 0.943 |
| Gradient Boosting | 0.216 | 0.275 | 0.935 |
| **Random Forest (tuned)** | **0.192** | **0.258** | **0.943** |

*(Metrics computed on log(GDP per capita); 5-fold CV R² = 0.923, confirming the result is robust,
not an artifact of one train/test split.)*

**Top predictors:** Electricity access (by far the strongest, importance ≈ 0.60), followed by
secondary school enrollment and unemployment.

## Key Findings & Recommendations

- GDP per capita **level** is highly predictable from development indicators; GDP **growth rate**
  is not — an important distinction for anyone considering indicator-based economic forecasting.
- **Electricity access is the dominant signal** of economic development level across African
  economies in this dataset — more so than health or investment indicators.
- Non-linear models (Random Forest, Gradient Boosting) meaningfully outperform Linear Regression,
  indicating threshold/diminishing-returns effects rather than simple linear relationships.
- **For policymakers:** electrification infrastructure investment aligns most closely with higher
  measured economic development in this data (correlation, not proven causation).
- **For future work:** incorporating governance and commodity-price data could improve
  growth-rate prediction specifically, since these better capture shock-driven variation.

## Repository Structure

```
├── notebook/
│   ├── Africa_GDP_Capstone_Project.ipynb   # Full analysis notebook (downloads/reads WDICSV.csv)
│   ├── africa_wdi_clean.csv                 # Cleaned, filtered dataset (ready to use)
│   ├── model_performance_comparison.csv     # Before/after model comparison
│   └── figures/                              # Saved plots
├── deploy/
│   ├── app.py                 # FastAPI prediction API
│   ├── predict.py              # Shared preprocessing + prediction logic
│   ├── streamlit_app.py        # Streamlit web UI
│   ├── test_api.py              # API test suite
│   ├── requirements.txt         # Python dependencies
│   └── model/                   # Saved model, scaler, feature list
├── report/
│   └── Capstone_Project_Report.pdf   # Full written project report
└── README.md
```

## Setup & Running

```bash
# 1. Install dependencies
cd deploy
pip install -r requirements.txt

# 2. Run the API
uvicorn app:app --reload --port 8002
# Interactive docs: http://127.0.0.1:8002/docs

# 3. Run the tests (with the API running in another terminal)
python test_api.py

# 4. Run the Streamlit app
streamlit run streamlit_app.py
# Open http://localhost:8501 in your browser
```

## API Reference

**POST** `/predict`

```json
{
  "Life_expectancy_years": 63.5,
  "Under5_mortality_per1000": 45.0,
  "Secondary_school_enrollment_pct": 55.0,
  "Education_expenditure_pct_govt": 15.0,
  "Gross_capital_formation_pct_gdp": 22.0,
  "FDI_net_inflows_usd": 500000000,
  "Electricity_access_pct": 65.0,
  "Population_growth_pct": 2.3,
  "Inflation_pct": 6.5,
  "Unemployment_pct": 9.0
}
```

**Response:**
```json
{
  "predicted_gdp_per_capita_usd": 2016.56,
  "predicted_log_gdp_per_capita": 7.6096
}
```

## Data Source

World Bank World Development Indicators (WDI) — official bulk download from
[datatopics.worldbank.org](https://datatopics.worldbank.org/world-development-indicators/)
(`WDI_CSV.zip`, containing `WDICSV.csv` and `WDICountry.csv`), filtered to Africa and the
2000–2020 window. License: World Bank Group Terms of Use (CC-BY-4.0).

## Acknowledgments

Built as the final Capstone Project for the Data Science Internship program at
**AnalystLab Africa**.

`#AnalystLabAfrica`
