# Model Benchmark Reportv1

Target: `Historical_Cost_of_Ride`

Transformation: None
## Summary Metrics

| Model | MAE | RMSE | R2 | BP p-value (F) | Tuned Params |
|---|---:|---:|---:|---:|---|
| ridge | 52.491 | 67.435 | 0.875 | 1.397e-10 | {'alpha': 1.0} |
| lasso | 52.092 | 67.564 | 0.875 | 2.833e-10 | {'alpha': 1.0} |
| gbrt | 54.107 | 70.625 | 0.863 | 8.06e-09 | {'learning_rate': 0.01, 'max_depth': 2, 'n_estimators': 400, 'subsample': 0.8} |
| random_forest | 54.404 | 72.665 | 0.855 | 5.416e-09 | {'n_estimators': 400, 'min_samples_leaf': 10, 'max_features': None, 'max_depth': None} |
| decision_tree | 56.552 | 74.055 | 0.850 | 1.22e-10 | {'max_depth': 5, 'min_samples_leaf': 1} |

## Diagnostics

Figures saved in `reports/figures/v1`:

- `qqplot_<model>.png`
- `resid_vs_fitted_<model>.png`
- `resid_hist_<model>.png`
