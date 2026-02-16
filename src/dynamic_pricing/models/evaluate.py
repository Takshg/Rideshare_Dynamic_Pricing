from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def qqplot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    model_name: str,
    output_dir: str | Path = "reports/figures", ) -> Path:
    
    residuals = y_true - y_pred
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"QQ Plot of Residuals — {model_name}")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.tight_layout()

    out_path = output_dir / f"qqplot_{model_name}.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def residuals_vs_fitted_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    model_name: str,
    output_dir: str | Path = "reports/figures",) -> Path:
    residuals = y_true - y_pred
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0.0)
    plt.title(f"Residuals vs Fitted — {model_name}")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.tight_layout()

    out_path = output_dir / f"resid_vs_fitted_{model_name}.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def residual_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    model_name: str,
    bins: int = 40,
    output_dir: str | Path = "reports/figures", ) -> Path:
    
    residuals = y_true - y_pred
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.hist(residuals, bins=bins)
    plt.title(f"Residual Histogram — {model_name}")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()

    out_path = output_dir / f"resid_hist_{model_name}.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def breusch_pagan_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray, ) -> Dict[str, float]:
    """
    Breusch–Pagan test for heteroskedasticity.
    Returns p-values and test statistics.
    Note: X should be the design matrix used by the model (after preprocessing).
    """
    residuals = y_true - y_pred
    X_const = sm.add_constant(X, has_constant="add")
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(residuals, X_const)
    return {
        "lm_stat": float(lm_stat),
        "lm_pvalue": float(lm_pvalue),
        "f_stat": float(f_stat),
        "f_pvalue": float(f_pvalue),
    }

