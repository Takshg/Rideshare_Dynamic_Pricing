# Model Benchmark Reportv2

Target: `Historical_Cost_of_Ride`

Transformation: Log
## Summary Metrics

| Model | MAE | RMSE | R2 | BP p-value (F) | Tuned Params |
|---|---:|---:|---:|---:|---|
| lasso | 52.337 | 67.201 | 0.876 | 5.349e-11 | {'alpha': 0.01} |
| ridge | 52.656 | 68.498 | 0.871 | 1.079e-09 | {'alpha': 1.0} |
| gbrt | 53.257 | 69.650 | 0.867 | 4.373e-09 | {'learning_rate': 0.01, 'max_depth': 2, 'n_estimators': 800, 'subsample': 0.8} |
| random_forest | 54.913 | 73.057 | 0.854 | 6.062e-09 | {'n_estimators': 700, 'min_samples_leaf': 2, 'max_features': None, 'max_depth': 6} |
| decision_tree | 57.051 | 75.049 | 0.846 | 7.567e-11 | {'max_depth': 5, 'min_samples_leaf': 1} |

## Diagnostics

Figures saved in `reports/figures/v2`:

- `qqplot_<model>.png`
- `resid_vs_fitted_<model>.png`
- `resid_hist_<model>.png`
