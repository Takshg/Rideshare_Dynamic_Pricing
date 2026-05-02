# Model Explainability

## Overview

This phase introduces explainability into the dynamic pricing system to understand how different features influence predicted ride prices.

The goal is to move beyond model performance metrics and provide **transparent, interpretable insights** into:

- Why a specific price is predicted
- Which features drive pricing decisions
- How different contexts impact pricing behavior

Explainability is critical for pricing systems, where decisions must be:

- Interpretable
- Auditable
- Defensible

---

## Method Used: SHAP (SHapley Additive exPlanations)

We use SHAP to explain model predictions.

SHAP is based on game theory and assigns each feature a contribution value representing how much it influences the prediction.

### Why SHAP?

- Consistent and theoretically grounded
- Works with any model (model-agnostic)
- Provides both:
  - Global explanations (feature importance)
  - Local explanations (individual predictions)

---

## Model Explained

The explainability analysis is applied to:

- **Model**: Ridge Regression (v1)
- **Feature Pipeline**: Engineered features + encoding + scaling
- **Artifacts**:
  - `artifacts/model/v1/model_bundle.joblib`

Ridge was selected due to:

- Strong predictive performance
- Stability under multicollinearity
- Simpler interpretation compared to tree-based models

---

## Explanation Pipeline

The process follows: Raw Data → FeaturePipeline → Model → SHAP Explainer → SHAP Values

Key design decision:

- SHAP is applied **after feature transformation**, ensuring explanations match the model’s actual input space.

---

## Outputs

SHAP analysis produces:

### 1. Summary Beeswarm Plot

File: reports/figures/shap/shap_summary_beeswarm.png

Shows:

- Feature importance
- Direction of impact
- Distribution of effects

---

### 2. Feature Importance Bar Plot

File: reports/figures/shap/shap_feature_importance_bar.png

Shows:

- Global feature importance ranking
- Mean absolute SHAP values

---

### 3. Feature Importance Table

File: reports/figures/shap/shap_feature_importance.csv

Contains:

- Feature names
- Mean absolute contribution values

---

### 4. Summary Report

File: reports/shap_summary.md


Includes:

- Top contributing features
- Links to generated plots

---

## Interpretation

### Global Insights

SHAP allows identification of the most influential features across all predictions.

Typical important features include:

- `Expected_Ride_Duration`
- `log_duration`
- `Vehicle_Type`
- `riders_per_driver`
- `Number_of_Riders`
- `Number_of_Drivers`

Interpretation example:

> Longer ride durations and premium vehicle types increase predicted prices.

---

### Directional Effects

The SHAP beeswarm plot shows:

- Positive SHAP values → increase predicted price
- Negative SHAP values → decrease predicted price

Example:

> Higher rider-to-driver ratios tend to increase price due to demand pressure.

---

### Local Explanations (Per Ride)

SHAP can explain individual predictions:

Example:

> A high predicted price for a ride may be driven by:
>
> - Long expected duration
> - Premium vehicle type
> - High rider demand relative to driver supply

This enables case-by-case transparency.

---

## Why Explainability Matters in Pricing

Pricing systems directly affect users and revenue.

Explainability enables:

- Trust in pricing decisions
- Detection of unintended biases
- Better business understanding of pricing drivers
- Debugging model behavior

---

## Limitations

1. SHAP explanations operate on transformed feature space

   - One-hot encoded features may be less intuitive
2. Correlated features can split importance across multiple variables
3. Explanations reflect the model’s learned relationships

   - Not necessarily causal relationships
4. SHAP values depend on the training data distribution

---

## Design Rationale

We chose SHAP because:

- It aligns with production-grade interpretability practices
- It supports both global and local explanations
- It works seamlessly with the existing pipeline and model bundle

---

## Future Improvements

- Add local explanation visualizations for individual rides
- Group one-hot encoded features for cleaner interpretation
- Compare explanations across v1 and v2 models
- Incorporate explanation-based validation (e.g., detecting anomalies)

---

## Conclusion

Explainability transforms the pricing system from a black-box model into a transparent decision-making tool.

It provides critical insight into:

- Feature importance
- Model behavior
- Pricing drivers

This strengthens the overall system by making it more interpretable, trustworthy, and actionable.
