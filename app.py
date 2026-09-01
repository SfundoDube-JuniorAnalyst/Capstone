"""
app.py — FastAPI prediction API for the Africa GDP per Capita model.

Run locally with:
    uvicorn app:app --reload --port 8002

Interactive docs available at http://127.0.0.1:8002/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import predict_gdp_per_capita, FEATURE_BOUNDS

app = FastAPI(
    title="Africa GDP per Capita Prediction API",
    description="Predicts a country's GDP per capita (US$) from health, education, "
                "investment, and infrastructure indicators, based on a Random Forest "
                "model trained on World Bank WDI data for 54 African countries (2000-2020).",
    version="1.0.0"
)


class IndicatorInput(BaseModel):
    Life_expectancy_years: float = Field(..., ge=30, le=90, description="Life expectancy at birth (years)")
    Under5_mortality_per1000: float = Field(..., ge=0, le=250, description="Under-5 mortality rate (per 1,000 live births)")
    Secondary_school_enrollment_pct: float = Field(..., ge=0, le=150, description="Secondary school enrollment (% gross)")
    Education_expenditure_pct_govt: float = Field(..., ge=0, le=50, description="Government education expenditure (% of govt spending)")
    Gross_capital_formation_pct_gdp: float = Field(..., ge=-20, le=80, description="Gross capital formation (% of GDP)")
    FDI_net_inflows_usd: float = Field(..., description="Foreign direct investment, net inflows (US$)")
    Electricity_access_pct: float = Field(..., ge=0, le=100, description="Access to electricity (% of population)")
    Population_growth_pct: float = Field(..., ge=-5, le=10, description="Population growth (annual %)")
    Inflation_pct: float = Field(..., ge=-20, le=500, description="Inflation, consumer prices (annual %)")
    Unemployment_pct: float = Field(..., ge=0, le=60, description="Unemployment (% of total labor force)")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionOutput(BaseModel):
    predicted_gdp_per_capita_usd: float
    predicted_log_gdp_per_capita: float


@app.get("/")
def root():
    return {
        "message": "Africa GDP per Capita Prediction API is running.",
        "docs": "/docs",
        "predict_endpoint": "/predict"
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/feature-bounds")
def feature_bounds():
    """Returns the plausible input range for each indicator."""
    return FEATURE_BOUNDS


@app.post("/predict", response_model=PredictionOutput)
def predict(indicators: IndicatorInput):
    try:
        result = predict_gdp_per_capita(indicators.model_dump())
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
