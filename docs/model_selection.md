
# Model Selection Decision Log

## Objective

Predict `Historical_Cost_of_Ride` using contextual and engineered features.

---

# Phase 1 — Baseline Modeling (v1: No Target Transform)

## Setup

- Feature engineering: ratios, differences, log features
- Models evaluated:
  - Ridge (L2)
  - Lasso (L1)
  - Decision Tree
  - Random Forest
  - Gradient Boosting (GBRT)
- Evaluation:
  - RMSE (primary metric)
  - R²
  - Residual diagnostics (QQ, residual vs fitted, histogram)
  - Breusch–Pagan test for heteroskedasticity

## v1 Results

From benchmark report:

| Model        | RMSE   | R²   |
| ------------ | ------ | ----- |
| Ridge        | 67.435 | 0.875 |
| Lasso        | 67.564 | 0.875 |
| GBRT         | 70.625 | 0.863 |
| RandomForest | 72.665 | 0.855 |
| DecisionTree | 74.055 | 0.850 |

(Full table available in reports/model_benchmark.md)

## v1 Diagnostic Findings

1. Residual vs fitted plots show a clear **funnel shape**.
   - Residual variance increases with predicted price.
2. Breusch–Pagan test strongly rejects homoscedasticity (p ≪ 0.001).
3. Ridge and Lasso perform best; tree models do not improve generalization.
4. Residual distribution approximately Gaussian in central region.

## v1 Decision

Ridge selected as baseline production candidate.

Rationale:

- Best RMSE.
- Stable under correlated engineered features.
- No evidence nonlinear models add structural benefit.

However:

Clear heteroskedasticity indicates variance increases with ride price.

This suggests the linear model assumptions are partially violated.

---

# Phase 2 — Log-Transformed Target (v2)

## Motivation

Heteroskedasticity in v1 suggests:

- Error variance scales with price magnitude.
- High-value rides produce larger absolute errors.

Statistical remedy:
Model log(price) instead of price.

This often stabilizes variance and improves residual structure.

---

## v2 Results (Log Transform)

| Model        | RMSE   | R²   |
| ------------ | ------ | ----- |
| Lasso        | 67.201 | 0.876 |
| Ridge        | 68.498 | 0.871 |
| GBRT         | 69.650 | 0.867 |
| RandomForest | 73.057 | 0.854 |
| DecisionTree | 75.049 | 0.846 |

(See benchmark report for full metrics.)

---

## v2 Diagnostic Findings

1. Residual vs fitted plots:
   - Slight reduction in extreme variance spread.
   - Funnel pattern still present but modestly reduced.
2. Breusch–Pagan test:
   - Still rejects homoscedasticity.
   - Indicates variance structure not fully eliminated.
3. Lasso slightly outperforms Ridge in RMSE under log transform.
4. Tree-based models still underperform linear models.

---

# Comparative Analysis: v1 vs v2

| Aspect             | v1             | v2                |
| ------------------ | -------------- | ----------------- |
| Best RMSE          | 67.435 (Ridge) | 67.201 (Lasso)    |
| Best R²           | 0.875          | 0.876             |
| Heteroskedasticity | Strong         | Slightly reduced  |
| Residual shape     | Funnel         | Slightly improved |
| Model ranking      | Linear > Trees | Linear > Trees    |

Observations:

- Log transform provides marginal RMSE improvement.
- Heteroskedasticity is reduced but not eliminated.
- Feature engineering already captured much structure.
- Linear models remain superior to tree-based methods.

---

# Final Production Decision

Selected Model: **Ridge (v1)**

Rationale:

1. Performance difference between v1 and v2 is marginal.
2. v1 is simpler and more interpretable.
3. Residual distribution is acceptable.
4. Variance instability does not materially degrade predictive accuracy.
5. Simpler deployment (no inverse-transform logic required).

v2 remains documented as an alternative configuration.

---

# Governance & Versioning

Artifacts saved as:

- artifacts/model/v1/ → baseline Ridge (price target)
- artifacts/model/v2/ → log-target variant

Versioning ensures:

- Reproducibility
- Traceable decision-making
- Future comparison capability

---

## Ridge vs Lasso Decision

Although Lasso achieved marginally higher R² under log-transform, the improvement was negligible (<0.5% relative difference). Given the presence of correlated engineered features, Ridge provides more stable coefficient shrinkage.
