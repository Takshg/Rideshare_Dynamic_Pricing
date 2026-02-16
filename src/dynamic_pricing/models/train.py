from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from dynamic_pricing.features.pipeline import FeaturePipeline
from dynamic_pricing.models.evaluate import (
    regression_metrics,
    qqplot_residuals,
    residuals_vs_fitted_plot,
    residual_histogram,
    breusch_pagan_test, )


@dataclass
class BenchmarkResult:
    model_name: str
    metrics: Dict[str, float]
    diagnostics: Dict[str, Any]
    model_obj: Any
    best_params: Dict[str, Any] | None = None


def make_models(random_state: int = 42) -> Dict[str, Any]:
    return {
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "lasso": Lasso(alpha=0.05, random_state=random_state, max_iter=50000, tol= 1e-3),
        "decision_tree": DecisionTreeRegressor(random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gbrt": GradientBoostingRegressor(random_state=random_state),
    }


def make_search_spaces(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Hyperparameter spaces for tuning.
    - GridSearchCV for smaller spaces
    - RandomizedSearchCV for larger spaces
    """
    return {
        "lasso": {
            "type": "grid",
            "params": {"alpha": [0.01, 0.05, 0.1, 0.5, 1.0]},
        },
        "decision_tree": {
            "type": "grid",
            "params": {"max_depth": [None, 3, 5, 8, 12], "min_samples_leaf": [1, 2, 5, 10]},
        },
        "random_forest": {
            "type": "random",
            "params": {
                "n_estimators": [200, 400, 700, 1000],
                "max_depth": [None, 6, 10, 16, 24],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": ["sqrt", "log2", None],
            },
            "n_iter": 20,
            "random_state": random_state,
        },
        "gbrt": {
            "type": "grid",
            "params": {
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [200, 400, 800],
                "max_depth": [2, 3, 4],
                "subsample": [0.8, 1.0],
            },
        },
        "ridge": {
            "type": "grid",
            "params": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        },
    }


def _fit_with_optional_tuning(
    name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    tune: bool,
    random_state: int,
    cv: int = 5,
    scoring: str = "neg_root_mean_squared_error", ) -> Tuple[Any, Dict[str, Any] | None]:
   
    if not tune:
        model.fit(X_train, y_train)
        return model, None

    spaces = make_search_spaces(random_state=random_state)
    spec = spaces.get(name)
    if spec is None:
        model.fit(X_train, y_train)
        return model, None

    if spec["type"] == "grid":
        search = GridSearchCV(model, spec["params"], cv=cv, scoring=scoring, n_jobs=-1)
    else:
        search = RandomizedSearchCV(
            model,
            spec["params"],
            n_iter=spec.get("n_iter", 20),
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=spec.get("random_state", random_state),
        )

    search.fit(X_train, y_train)
    return search.best_estimator_, dict(search.best_params_)


def benchmark_models(
    df: pd.DataFrame,
    *,
    target_col: str = "Historical_Cost_of_Ride",
    test_size: float = 0.2,
    random_state: int = 42,
    figures_dir: str | Path = "reports/figures",
    tune: bool = True,
    best_params: dict | None = None,  
    target_transform: str = "log", ) -> Tuple[List[BenchmarkResult], FeaturePipeline]:
    
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    y_train_raw = train_df[target_col].to_numpy(dtype=float)
    y_test_raw = test_df[target_col].to_numpy(dtype=float)

    if target_transform == "log":
        # guard against log(0) although prices should be >0
        y_train = np.log(np.maximum(y_train_raw, 1e-8))
        y_test = np.log(np.maximum(y_test_raw, 1e-8))
    elif target_transform == "none":
        y_train = y_train_raw
        y_test = y_test_raw
    else:
        raise ValueError("target_transform must be 'none' or 'log'")
    
    fp = FeaturePipeline(feature_version="v2")
    X_train, feat_names = fp.fit_transform(train_df)
    X_test, _ = fp.transform(test_df)
    results: List[BenchmarkResult] = []


    for name, model in make_models(random_state=random_state).items():
        fitted, best_params = _fit_with_optional_tuning(
            name, model, X_train, y_train, tune=tune, random_state=random_state
        )

        preds = fitted.predict(X_test)
        # If trained in log space, convert predictions back to price space for evaluation/plots
        if target_transform == "log":
            preds_price = np.exp(preds)
            y_eval = y_test_raw
        else:
            preds_price = preds
            y_eval = y_test_raw
       
        metrics = regression_metrics(y_eval, preds_price)
        
        diagnostics: dict = {}
        
        # Plots
        qqplot_residuals(y_eval, preds_price, model_name=name, output_dir=figures_dir)
        residuals_vs_fitted_plot(y_eval, preds_price, model_name=name, output_dir=figures_dir)
        residual_histogram(y_eval, preds_price, model_name=name, output_dir=figures_dir)
        
        # Homoscedasticity test
        bp = breusch_pagan_test(y_eval, preds_price, X_test)
        diagnostics["breusch_pagan"] = bp

        results.append(
            BenchmarkResult(
                model_name=name,
                metrics=metrics,
                diagnostics=diagnostics,
                model_obj=fitted,
                best_params=best_params,
            )
        )
    results.sort(key=lambda r: r.metrics["rmse"])
    return results, fp