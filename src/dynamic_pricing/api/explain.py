from __future__ import annotations

import numpy as np
import pandas as pd


def explain_linear_prediction(bundle, df: pd.DataFrame, *, top_n: int = 10) -> dict:
    """
    Lightweight local explanation for linear models.

    For Ridge/Lasso:
    contribution_j = transformed_feature_j * coefficient_j

    This explains the prediction in the exact transformed feature space used by the model.
    """
    X, feature_names = bundle.feature_pipeline.transform(df)
    model = bundle.model

    if not hasattr(model, "coef_"):
        raise TypeError("Explanation endpoint currently supports linear models with coef_ only.")

    coef = np.asarray(model.coef_).ravel()
    x = np.asarray(X[0]).ravel()

    contributions = x * coef

    ranked_idx = np.argsort(np.abs(contributions))[::-1][:top_n]

    top = [
        {
            "feature": feature_names[i],
            "value": float(x[i]),
            "contribution": float(contributions[i]),
        }
        for i in ranked_idx
    ]

    prediction = float(model.predict(X)[0])

    return {
        "predicted_price": prediction,
        "top_contributions": top,
        "explanation_note": (
            "Local linear explanation computed as transformed feature value × model coefficient. "
            "Contributions are shown in the model's transformed feature space."
        ),
    }