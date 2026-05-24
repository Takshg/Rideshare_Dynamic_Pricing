# Evaluation and Improvement Metrics

This report summarizes model, pricing-policy, and explainability improvement metrics.

## 1. Model Improvement Metrics

The naive baseline predicts the mean historical ride price from the training split. Model improvement is measured against this baseline.

| version | model_name | target_transform | mae | rmse | r2 | mae_reduction_vs_baseline_pct | rmse_reduction_vs_baseline_pct | r2_lift_vs_baseline | rmse_delta_vs_v1 | mae_delta_vs_v1 | r2_delta_vs_v1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | mean_price_baseline | none | 160.9642 | 191.1528 | -0.0022 | 0.0000 | 0.0000 | 0.0000 | - | - | - |
| v1 | ridge | none | 52.4907 | 67.4353 | 0.8753 | 67.3898 | 64.7218 | 0.8774 | - | - | - |
| v2 | lasso | log | 52.3371 | 67.2008 | 0.8761 | 67.4852 | 64.8445 | 0.8783 | -0.2345 | -0.1536 | 0.0009 |

## 2. Pricing Policy Improvement Metrics

Policy improvements are computed from the counterfactual simulation outputs. Revenue lift is measured relative to the Historical policy when available.

| policy | avg_price | avg_price_delta_pct | avg_acceptance_probability | expected_accepted_rides | total_expected_revenue | avg_expected_revenue | price_std | surge_frequency_25pct | extreme_surge_frequency_50pct | customer_pain_index_pct | expected_revenue_lift_vs_historical_pct | acceptance_delta_vs_historical | price_delta_vs_historical_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Historical | 372.5026 | 0.0000 | 0.5000 | 500.0000 | 186,251.31 | 186.2513 | 187.1588 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| demand_supply | 612.9152 | 64.7995 | 0.1362 | 136.1753 | 70,140.45 | 70.1404 | 330.9262 | 81.7000 | 60.4000 | 64.7995 | -62.3410 | -0.3638 | 64.5398 |
| ridge_model | 373.3830 | 3.2560 | 0.4756 | 475.6417 | 177,738.96 | 177.7390 | 175.0299 | 13.4000 | 0.9000 | 9.0760 | -4.5704 | -0.0244 | 0.2364 |

## 3. Explainability Metrics

These metrics summarize how concentrated the SHAP importance distribution is.

| top_feature | top_feature_mean_abs_shap | top_5_feature_coverage_pct | top_1_feature_share_pct | num_features_explained |
| --- | --- | --- | --- | --- |
| Expected_Ride_Duration | 162.4440 | 88.3150 | 72.7764 | 21 |

## Notes

- Simulation metrics are based on a heuristic probabilistic acceptance model, not observed acceptance labels.
- SHAP metrics explain model behavior, not causal effects.
- v2 metrics are back-transformed to original price scale when target_transform is log.
