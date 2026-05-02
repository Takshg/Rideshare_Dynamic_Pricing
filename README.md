# Rideshare Dynamic Pricing System

## Taksh Girdhar

An end-to-end dynamic pricing system for rideshare platforms, covering:

- Data validation and feature engineering
- Model benchmarking and diagnostics
- Model versioning and artifact management
- Counterfactual pricing policy simulation
- Probabilistic customer acceptance modeling
- SHAP-based explainability
- FastAPI model serving
- Logging and Prometheus-style monitoring
- Docker containerization
- Kubernetes-ready deployment configuration

This project is structured as a production-oriented machine learning system rather than a notebook-only analysis.

---

## Project Report

A detailed project report is available in:

```text
Project Report.pdf
```

The report explains the complete system design, methodology, results, diagnostics, simulation assumptions, explainability layer, API deployment, Docker setup, Kubernetes-ready architecture, limitations, and future work.

It is the best place to start if you want a full narrative explanation of the project.

Key report sections include:

- Dataset and problem definition
- Feature engineering pipeline
- Model benchmarking and selection
- v1 vs v2 model comparison
- Counterfactual pricing simulation
- SHAP explainability
- FastAPI API deployment
- Docker and Kubernetes deployment
- Key results, limitations, and future extensions

---

## Project Summary

This project predicts rideshare prices using contextual ride features such as demand, supply, customer information, booking time, vehicle type, and ride duration.

The final selected production model is:

```text
Ridge Regression v1
```

Ridge v1 was selected because it provides:

- Strong predictive performance
- Stable coefficient shrinkage under correlated engineered features
- Interpretability
- Simpler deployment compared with the log-transformed v2 model

Although the v2 log-target experiment achieved a marginally lower RMSE with Lasso, the improvement was small and did not justify the additional deployment complexity.

---

## System Architecture

The system follows this high-level flow:

```text
Raw rideshare data
        ↓
CSV validation + FeaturePipeline
        ↓
Model benchmarking and diagnostics
        ↓
Versioned ModelBundle artifact
        ↓
Simulation + Explainability + API serving
        ↓
Docker + Kubernetes-ready deployment
```

Main deployed API flow:

```text
API request
   ↓
Pydantic validation
   ↓
Saved FeaturePipeline transformation
   ↓
Ridge v1 model prediction
   ↓
Price response and optional explanation
```

---

## Repository Structure

```text
Rideshare_Dynamic_Pricing/
├─ src/
│  └─ dynamic_pricing/
│     ├─ api/
│     │  ├─ main.py              # FastAPI app entrypoint
│     │  ├─ routes.py            # API routes: /health, /predict, /explain, /metrics
│     │  ├─ deps.py              # ModelBundle loader
│     │  ├─ explain.py           # Local linear explanation logic
│     │  └─ middleware.py        # Logging + Prometheus metrics middleware
│     │
│     ├─ config/
│     │  ├─ settings.py          # Environment-driven settings
│     │  └─ logging.py           # Logging configuration
│     │
│     ├─ schemas/
│     │  ├─ api.py               # API request/response schemas
│     │  ├─ features.py          # Ride context schema
│     │  └─ responses.py         # Prediction response schemas
│     │
│     ├─ features/
│     │  ├─ io.py                # CSV loading + schema validation
│     │  └─ pipeline.py          # FeaturePipeline: scaling, encoding, derived features
│     │
│     ├─ models/
│     │  ├─ bundle.py            # ModelBundle dataclass
│     │  ├─ evaluate.py          # Metrics and residual diagnostics
│     │  ├─ registry.py          # Save/load model artifacts
│     │  └─ train.py             # Model benchmarking + tuning
│     │
│     ├─ simulation/
│     │  ├─ acceptance.py        # Probabilistic acceptance model
│     │  ├─ policies.py          # Historical, Ridge, demand-supply pricing policies
│     │  └─ simulator.py         # Counterfactual policy simulation engine
│     │
│     └─ explainability/
│        └─ shap_analysis.py     # Offline SHAP analysis
│
├─ scripts/
│  ├─ smoke_features.py          # Feature pipeline smoke test
│  ├─ train_benchmark.py         # Train v1/v2 benchmark models
│  ├─ run_simulation.py          # Run pricing policy simulation
│  └─ run_shap_analysis.py       # Generate SHAP outputs
│
├─ docs/
│  ├─ data_dictionary.md
│  ├─ feature_pipeline.md
│  ├─ model_selection.md
│  ├─ simulation_methodology.md
│  └─ explainability.md
│
├─ reports/
│  ├─ model_benchmark_v1.md
│  ├─ model_benchmark_v2.md
│  ├─ simulation_results.md
│  ├─ shap_summary.md
│  └─ figures/
│     ├─ v1/
│     ├─ v2/
│     └─ shap/
│
├─ artifacts/
│  └─ model/
│     ├─ v1/
│     │  ├─ model_bundle.joblib
│     │  └─ metadata.json
│     └─ v2/
│        ├─ model_bundle.joblib
│        └─ metadata.json
│
├─ deploy/
│  └─ k8s/
│     └─ base/
│        ├─ deployment.yaml
│        ├─ service.yaml
│        └─ configmap.yaml
│
├─ Dockerfile
├─ docker-compose.yml
├─ pyproject.toml
├─ Project Report.pdf
└─ README.md
```

---

## Dataset

Place the raw dataset at:

```text
data/raw/dynamic_pricing.csv
```

### Required Columns

The CSV loader expects the following case-sensitive columns:

```text
Number_of_Riders
Number_of_Drivers
Location_Category
Customer_Loyalty_Status
Number_of_Past_Rides
Average_Ratings
Time_of_Booking
Vehicle_Type
Expected_Ride_Duration
Historical_Cost_of_Ride
```

The loader performs:

- Strict column validation
- Header whitespace stripping
- Numeric type coercion
- Categorical whitespace trimming

---

## Feature Engineering Pipeline

The project uses a reusable `FeaturePipeline` class as the single source of truth for preprocessing.

The same pipeline is used for:

- Model training
- Simulation
- API inference
- Explainability

This prevents training-serving skew.

### Raw Numeric Features

```text
Number_of_Riders
Number_of_Drivers
Number_of_Past_Rides
Average_Ratings
Expected_Ride_Duration
```

Transformation:

```text
StandardScaler
```

### Derived Numeric Features

```text
riders_per_driver = Number_of_Riders / max(Number_of_Drivers, 1)
driver_supply_gap = Number_of_Drivers - Number_of_Riders
log_riders = log1p(Number_of_Riders)
log_drivers = log1p(Number_of_Drivers)
log_duration = log1p(Expected_Ride_Duration)
```

Transformation:

```text
StandardScaler
```

### Categorical Features

```text
Location_Category
Customer_Loyalty_Status
Time_of_Booking
Vehicle_Type
```

Transformation:

```text
OneHotEncoder(handle_unknown="ignore")
```

---

## Model Benchmarking

The following models were benchmarked:

- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Hyperparameter tuning was performed using:

- `GridSearchCV`
- `RandomizedSearchCV`

Evaluation metrics included:

- MAE
- RMSE
- R²
- QQ plots
- Residuals vs fitted plots
- Residual histograms
- Breusch-Pagan heteroskedasticity test

---

## Model Versions

### v1: Price Target

The v1 experiment directly predicts:

```text
Historical_Cost_of_Ride
```

Best model:

```text
Ridge Regression
```

Approximate benchmark result:

```text
RMSE: 67.435
R²:   0.875
```

Artifacts:

```text
artifacts/model/v1/
```

### v2: Log-Price Target

The v2 experiment trains models on:

```text
log(Historical_Cost_of_Ride)
```

Predictions are back-transformed for evaluation.

Best model:

```text
Lasso Regression
```

Approximate benchmark result:

```text
RMSE: 67.201
R²:   0.876
```

Artifacts:

```text
artifacts/model/v2/
```

### Final Model Decision

The production candidate is:

```text
Ridge Regression v1
```

Rationale:

- v2 only marginally improved RMSE
- Ridge v1 is simpler to deploy
- Ridge is stable under correlated engineered features
- No inverse target transformation is required
- Residual diagnostics were acceptable for the project objective

See:

```text
docs/model_selection.md
reports/Project Report.pdf
```

---

## Model Artifacts

Models are saved as `ModelBundle` objects using `joblib`.

Each bundle contains:

- Trained model
- Fitted `FeaturePipeline`
- Model name
- Model version
- Feature version
- Metadata

Example:

```text
artifacts/model/v1/
├─ model_bundle.joblib
└─ metadata.json
```

This ensures API inference uses the exact same preprocessing pipeline as training.

---

## Counterfactual Pricing Simulation

The project includes a simulation framework that compares pricing policies.

Implemented policies:

1. `HistoricalPolicy`

   - Uses the original historical ride price
2. `RidgeModelPolicy`

   - Uses the selected Ridge v1 model to predict price
3. `DemandSupplyPolicy`

   - Applies a rule-based multiplier using rider/driver imbalance

Simulation flow:

```text
Ride context
   ↓
Pricing policy
   ↓
Proposed price
   ↓
Acceptance model
   ↓
Expected revenue
```

### Acceptance Probability Model

Since the dataset does not contain true acceptance/rejection labels, customer acceptance is simulated using a logistic function:

```text
P(accept) = 1 / (1 + exp(k * delta))
```

where:

```text
delta = (proposed_price - base_price) / base_price
```

Expected revenue is then:

```text
Expected Revenue = Proposed Price × P(accept)
```

Simulation outputs:

```text
reports/simulation_results.md
reports/simulation_detail_Historical.csv
reports/simulation_detail_ridge_model.csv
reports/simulation_detail_demand_supply.csv
```

Important limitation:

The simulation is not causal. It is a controlled policy comparison framework, not a real-world revenue forecast.

See:

```text
docs/simulation_methodology.md
```

---

## Explainability

Explainability is implemented in two forms:

### 1. Offline SHAP Analysis

The project generates SHAP-based global explanations for the selected Ridge v1 model.

Outputs:

```text
reports/figures/shap/shap_feature_importance_bar.png
reports/figures/shap/shap_summary_beeswarm.png
reports/figures/shap/shap_feature_importance.csv
reports/shap_summary.md
```

The top feature importance results show that `Expected_Ride_Duration` is the dominant driver of predicted price, followed by vehicle type and demand-related engineered features.

### 2. Real-Time API Explanation

The FastAPI service includes:

```text
POST /explain
```

For linear models, local contribution is computed as:

```text
contribution_j = transformed_feature_j × coefficient_j
```

This returns the top feature contributions for an individual prediction.

See:

```text
docs/explainability.md
```

---

## FastAPI Service

The model is deployed through a FastAPI service.

### Endpoints

```text
GET  /health
GET  /metrics
POST /predict
POST /explain
```

### `/health`

Checks whether the service is running.

### `/predict`

Returns predicted ride price.

### `/explain`

Returns predicted ride price plus top local feature contributions.

### `/metrics`

Returns Prometheus-compatible metrics, including:

- Request count by method, endpoint, and status
- Request latency histogram

---

## Example API Request

```json
{
  "Number_of_Riders": 60,
  "Number_of_Drivers": 25,
  "Location_Category": "Urban",
  "Customer_Loyalty_Status": "Gold",
  "Number_of_Past_Rides": 10,
  "Average_Ratings": 4.2,
  "Time_of_Booking": "Evening",
  "Vehicle_Type": "Economy",
  "Expected_Ride_Duration": 40,
  "Historical_Cost_of_Ride": 200.0
}
```

Example `/predict` response:

```json
{
  "predicted_price": 132.66,
  "model_name": "ridge",
  "model_version": "v1",
  "feature_version": "v1"
}
```

Example `/explain` response:

```json
{
  "predicted_price": 132.66,
  "model_name": "ridge",
  "model_version": "v1",
  "feature_version": "v1",
  "top_contributions": [
    {
      "feature": "Expected_Ride_Duration",
      "value": -1.2066,
      "contribution": -215.1185
    },
    {
      "feature": "Vehicle_Type_Economy",
      "value": 1.0,
      "contribution": -21.0976
    }
  ]
}
```

---

## Logging and Monitoring

The API includes request logging middleware.

Each request logs:

- HTTP method
- Endpoint path
- Response status
- Request duration

Prometheus-style metrics are available at:

```text
GET /metrics
```

Tracked metrics include:

- `dynamic_pricing_http_requests_total`
- `dynamic_pricing_http_request_duration_seconds`

---

## Docker

The project includes a Dockerfile for containerized API serving.

Build the image:

```bash
docker build -t dynamic-pricing-api:v1 .
```

Run the container:

```bash
docker run --rm -p 8000:8000 dynamic-pricing-api:v1
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Docker Compose

Run locally with Docker Compose:

```bash
docker compose up --build
```

The API will be available at:

```text
http://127.0.0.1:8000/docs
```

---

## Kubernetes-Ready Deployment

Kubernetes manifests are stored in:

```text
deploy/k8s/base/
```

Included manifests:

```text
deployment.yaml
service.yaml
configmap.yaml
```

The Kubernetes setup defines:

- FastAPI deployment
- Multiple API replicas
- ClusterIP service
- ConfigMap for model/service configuration

This prepares the system for deployment using Minikube, kind, or a cloud Kubernetes cluster.

---

## Setup

### 1. Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install project

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

For API dependencies:

```bash
pip install -e ".[api]"
```

For explainability dependencies:

```bash
pip install -e ".[explain]"
```

Or install all optional dependencies:

```bash
pip install -e ".[dev,api,explain]"
```

---

## Common Commands

### Validate feature pipeline

```bash
python scripts/smoke_features.py
```

### Train v1 benchmark

```bash
python scripts/train_benchmark.py --version v1
```

### Train v2 benchmark

```bash
python scripts/train_benchmark.py --version v2
```

### Run pricing simulation

```bash
python scripts/run_simulation.py
```

### Run SHAP analysis

```bash
python scripts/run_shap_analysis.py
```

### Run FastAPI locally

```bash
uvicorn dynamic_pricing.api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

### Run tests

```bash
pytest
```

---

## Reports and Outputs

Important generated outputs:

```text
reports/Project Report.pdf
reports/model_benchmark_v1.md
reports/model_benchmark_v2.md
reports/simulation_results.md
reports/shap_summary.md
```

Important figures:

```text
reports/figures/v1/
reports/figures/v2/
reports/figures/shap/
```

Important artifacts:

```text
artifacts/model/v1/
artifacts/model/v2/
```

---

## Documentation

Detailed documentation is available in:

```text
docs/data_dictionary.md
docs/feature_pipeline.md
docs/model_selection.md
docs/simulation_methodology.md
docs/explainability.md
```

Recommended reading order:

1. `reports/Project Report.pdf`
2. `docs/feature_pipeline.md`
3. `docs/model_selection.md`
4. `docs/simulation_methodology.md`
5. `docs/explainability.md`

---

## Current Status

Completed:

- Data validation
- Feature engineering pipeline
- Model benchmarking
- Hyperparameter tuning
- Residual diagnostics
- v1/v2 model versioning
- Ridge v1 production model selection
- Counterfactual pricing simulation
- Probabilistic acceptance model
- SHAP explainability
- FastAPI deployment
- `/predict` endpoint
- `/explain` endpoint
- `/metrics` endpoint
- Request logging
- Docker containerization
- Docker Compose setup
- Kubernetes-ready manifests

---

## Limitations

- The dataset does not include true customer acceptance or rejection outcomes.
- The simulation acceptance model is heuristic, not learned from observed conversion data.
- The pricing model predicts historical price rather than directly optimizing profit.
- The system does not currently model long-term customer behavior, driver response, competition, or retention effects.
- SHAP and local feature contributions explain model behavior, not causal relationships.

---

## Design Philosophy

This project prioritizes:

- Reproducibility
- Versioned artifacts
- Strict data contracts
- Training-serving consistency
- Transparent model selection
- Explainability
- Deployment readiness
- Clear separation of concerns

The result is a complete applied machine learning system that goes beyond prediction and includes simulation, interpretability, monitoring, and deployment.
