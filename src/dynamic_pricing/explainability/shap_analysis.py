from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap


def run_shap_analysis(
    model_bundle,
    df: pd.DataFrame,
    *,
    output_dir: str | Path = "reports/figures/shap",
    max_rows: int = 500,
) -> dict:
    """
    Runs SHAP analysis for the trained model bundle.

    Uses the saved FeaturePipeline to transform raw data, ensuring
    explanations match the same feature space used by training.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fp = model_bundle.feature_pipeline
    model = model_bundle.model

    sample_df = df.sample(
        n=min(max_rows, len(df)),
        random_state=42,
    )

    X, feature_names = fp.transform(sample_df)

    explainer = shap.Explainer(model, X, feature_names=feature_names)
    shap_values = explainer(X)

    # Summary beeswarm
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    summary_path = output_dir / "shap_summary_beeswarm.png"
    plt.tight_layout()
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()

    # Bar plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    bar_path = output_dir / "shap_feature_importance_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close()

    mean_abs = abs(shap_values.values).mean(axis=0)

    importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    importance_path = output_dir / "shap_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    return {
        "summary_plot": str(summary_path),
        "bar_plot": str(bar_path),
        "importance_csv": str(importance_path),
        "top_features": importance.head(15).to_dict(orient="records"),
    }