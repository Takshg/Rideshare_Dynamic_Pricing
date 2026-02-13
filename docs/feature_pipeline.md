
# Feature Pipeline Specification (v1)

This document defines the feature engineering contract for the dynamic pricing system.

The `FeaturePipeline` is the single source of truth for:

- Model training
- Simulation
- API inference (Phase 6)
- Explainability (Phase 5)

---

# Feature Version

Current version: **v1**

Any modification to:

- Feature definitions
- Feature ordering
- Derived calculations
- Encoding logic

Requires:

- Incrementing `feature_version`
- Re-saving artifacts
- Re-training downstream models

---

# Raw Numeric Features

The following columns are passed directly into numeric preprocessing:

- `Number_of_Riders`
- `Number_of_Drivers`
- `Number_of_Past_Rides`
- `Average_Ratings`
- `Expected_Ride_Duration`

Preprocessing:

- StandardScaler (zero mean, unit variance)

---

# Derived Numeric Features

Derived features are computed before scaling.

### 1. riders_per_driver

Number_of_Riders / max(Number_of_Drivers, 1)

Purpose: Demand pressure ratio.

---

### 2. driver_supply_gap

Number_of_Drivers - Number_of_Riders

Purpose: Supply imbalance indicator.

---

### 3. log_riders

log1p(Number_of_Riders)

Purpose: Stabilizes skewed demand distribution.

---

### 4. log_drivers

log1p(Number_of_Drivers)

Purpose: Stabilizes supply skew.

---

### 5. log_duration

log1p(Expected_Ride_Duration)

Purpose: Handles long-tail trip durations.

---

# Categorical Features

Categorical features are one-hot encoded using:


Features:

- `Location_Category`
- `Customer_Loyalty_Status`
- `Time_of_Booking`
- `Vehicle_Type`

Unknown categories at inference:

- Ignored (no crash)
- Result in zero vector for that category

---

# ColumnTransformer Structure

The pipeline uses:

- Numeric → StandardScaler
- Categorical → OneHotEncoder
- Remainder → Drop

Output:

- Dense NumPy matrix `X`
- Ordered feature names available via `feature_names_`

---

# Artifact Contract

Saved under:

```
artifacts/feature_pipeline/
├─ feature_pipeline.joblib
└─ metadata.json
```

Metadata includes:

- feature_version
- training data shape
- creation timestamp

---

# Inference Requirements (Future API)

All inference requests must:

1. Match the original dataset schema.
2. Be transformed using the exact saved feature pipeline.
3. Not bypass preprocessing.

The API service will:

- Load `feature_pipeline.joblib`
- Transform incoming requests
- Pass features to the trained model

---

# Reproducibility Guarantee

Given:

- Same dataset
- Same feature version
- Same saved artifact

The pipeline will produce:

- Identical feature matrix shape
- Identical column ordering

This guarantees compatibility across:

- Training
- Simulation
- SHAP analysis
- FastAPI deployment
- Docker/Kubernetes runtime
