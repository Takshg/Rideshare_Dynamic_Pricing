from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from dynamic_pricing.features.io import load_csv_file
except ImportError:
    from dynamic_pricing.features.io import load_dynamic_pricing_csv as load_csv_file

from dynamic_pricing.models.registry import load_model_bundle


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def pct_change(new: float, old: float) -> float:
    if old == 0:
        return float("nan")
    return float(((new - old) / old) * 100)


def pct_reduction(old: float, new: float) -> float:
    if old == 0:
        return float("nan")
    return float(((old - new) / old) * 100)


def load_metadata(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def infer_target_transform(version: str, metadata: dict[str, Any]) -> str:
    if "target_transform" in metadata:
        return str(metadata["target_transform"])

    # Safe fallback based on project convention
    if version == "v2":
        return "log"
    return "none"


def predict_with_bundle(bundle: Any, df: pd.DataFrame, target_transform: str) -> np.ndarray:
    X, _ = bundle.feature_pipeline.transform(df)
    preds = np.asarray(bundle.model.predict(X), dtype=float)

    if target_transform == "log":
        preds = np.exp(preds)

    return np.maximum(preds, 0.0)


def compute_model_improvement_metrics(
    root: Path,
    *,
    versions: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    df = load_csv_file(root / "data" / "raw" / "dynamic_pricing.csv")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    target_col = "Historical_Cost_of_Ride"

    y_train = train_df[target_col].to_numpy(dtype=float)
    y_test = test_df[target_col].to_numpy(dtype=float)

    naive_pred = np.full_like(y_test, fill_value=float(np.mean(y_train)), dtype=float)
    naive_metrics = regression_metrics(y_test, naive_pred)

    rows: list[dict[str, Any]] = []

    rows.append(
        {
            "version": "baseline",
            "model_name": "mean_price_baseline",
            "target_transform": "none",
            "mae": naive_metrics["mae"],
            "rmse": naive_metrics["rmse"],
            "r2": naive_metrics["r2"],
            "mae_reduction_vs_baseline_pct": 0.0,
            "rmse_reduction_vs_baseline_pct": 0.0,
            "r2_lift_vs_baseline": 0.0,
        }
    )

    for version in versions:
        model_dir = root / "artifacts" / "model" / version
        bundle_path = model_dir / "model_bundle.joblib"

        if not bundle_path.exists():
            print(f"Skipping {version}: missing {bundle_path}")
            continue

        metadata = load_metadata(model_dir)
        target_transform = infer_target_transform(version, metadata)

        bundle = load_model_bundle(model_dir)
        preds = predict_with_bundle(bundle, test_df, target_transform)

        metrics = regression_metrics(y_test, preds)

        rows.append(
            {
                "version": version,
                "model_name": bundle.model_name,
                "target_transform": target_transform,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "mae_reduction_vs_baseline_pct": pct_reduction(
                    naive_metrics["mae"], metrics["mae"]
                ),
                "rmse_reduction_vs_baseline_pct": pct_reduction(
                    naive_metrics["rmse"], metrics["rmse"]
                ),
                "r2_lift_vs_baseline": metrics["r2"] - naive_metrics["r2"],
            }
        )

    model_df = pd.DataFrame(rows)

    # Add v2-v1 comparison if both exist
    if {"v1", "v2"}.issubset(set(model_df["version"])):
        v1 = model_df.loc[model_df["version"] == "v1"].iloc[0]
        v2 = model_df.loc[model_df["version"] == "v2"].iloc[0]

        model_df.loc[model_df["version"] == "v2", "rmse_delta_vs_v1"] = (
            float(v2["rmse"]) - float(v1["rmse"])
        )
        model_df.loc[model_df["version"] == "v2", "mae_delta_vs_v1"] = (
            float(v2["mae"]) - float(v1["mae"])
        )
        model_df.loc[model_df["version"] == "v2", "r2_delta_vs_v1"] = (
            float(v2["r2"]) - float(v1["r2"])
        )

    return model_df


def compute_policy_improvement_metrics(root: Path) -> pd.DataFrame:
    report_dir = root / "reports"
    detail_files = sorted(report_dir.glob("simulation_detail_*.csv"))

    rows: list[dict[str, Any]] = []

    for file in detail_files:
        sim = pd.read_csv(file)
        policy = file.stem.replace("simulation_detail_", "")

        required = {"base_price", "proposed_price", "acceptance_probability", "expected_revenue"}
        if not required.issubset(set(sim.columns)):
            print(f"Skipping {file.name}: missing required simulation columns.")
            continue

        if "price_delta_pct" not in sim.columns:
            sim["price_delta_pct"] = (
                sim["proposed_price"] - sim["base_price"]
            ) / sim["base_price"]

        total_expected_revenue = float(sim["expected_revenue"].sum())
        avg_acceptance = float(sim["acceptance_probability"].mean())
        expected_accepted_rides = float(sim["acceptance_probability"].sum())

        rows.append(
            {
                "policy": policy,
                "avg_price": float(sim["proposed_price"].mean()),
                "avg_price_delta_pct": float(sim["price_delta_pct"].mean() * 100),
                "avg_acceptance_probability": avg_acceptance,
                "expected_accepted_rides": expected_accepted_rides,
                "total_expected_revenue": total_expected_revenue,
                "avg_expected_revenue": float(sim["expected_revenue"].mean()),
                "price_std": float(sim["proposed_price"].std()),
                "surge_frequency_25pct": float(
                    (sim["proposed_price"] > 1.25 * sim["base_price"]).mean() * 100
                ),
                "extreme_surge_frequency_50pct": float(
                    (sim["proposed_price"] > 1.50 * sim["base_price"]).mean() * 100
                ),
                "customer_pain_index_pct": float(
                    np.maximum(sim["price_delta_pct"], 0).mean() * 100
                ),
            }
        )

    policy_df = pd.DataFrame(rows)

    if policy_df.empty:
        return policy_df

    # Find baseline policy
    baseline_candidates = policy_df[
        policy_df["policy"].str.lower().str.contains("historical")
    ]

    if not baseline_candidates.empty:
        baseline = baseline_candidates.iloc[0]

        policy_df["expected_revenue_lift_vs_historical_pct"] = policy_df[
            "total_expected_revenue"
        ].apply(lambda x: pct_change(float(x), float(baseline["total_expected_revenue"])))

        policy_df["acceptance_delta_vs_historical"] = (
            policy_df["avg_acceptance_probability"]
            - float(baseline["avg_acceptance_probability"])
        )

        policy_df["price_delta_vs_historical_pct"] = policy_df["avg_price"].apply(
            lambda x: pct_change(float(x), float(baseline["avg_price"]))
        )

    return policy_df


def compute_explainability_metrics(root: Path, *, top_k: int = 5) -> pd.DataFrame:
    shap_path = root / "reports" / "figures" / "shap" / "shap_feature_importance.csv"

    if not shap_path.exists():
        print(f"Skipping SHAP metrics: missing {shap_path}")
        return pd.DataFrame()

    shap_df = pd.read_csv(shap_path)

    if not {"feature", "mean_abs_shap"}.issubset(set(shap_df.columns)):
        print("Skipping SHAP metrics: expected columns feature and mean_abs_shap.")
        return pd.DataFrame()

    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    total_importance = float(shap_df["mean_abs_shap"].sum())
    top_k_importance = float(shap_df.head(top_k)["mean_abs_shap"].sum())
    top_1_importance = float(shap_df.head(1)["mean_abs_shap"].sum())

    if total_importance == 0:
        top_k_coverage = float("nan")
        top_1_share = float("nan")
    else:
        top_k_coverage = float((top_k_importance / total_importance) * 100)
        top_1_share = float((top_1_importance / total_importance) * 100)

    return pd.DataFrame(
        [
            {
                "top_feature": str(shap_df.iloc[0]["feature"]),
                "top_feature_mean_abs_shap": float(shap_df.iloc[0]["mean_abs_shap"]),
                f"top_{top_k}_feature_coverage_pct": top_k_coverage,
                "top_1_feature_share_pct": top_1_share,
                "num_features_explained": int(len(shap_df)),
            }
        ]
    )


def format_value(x: Any) -> str:
    if pd.isna(x):
        return "-"
    if isinstance(x, float):
        if abs(x) >= 1000:
            return f"{x:,.2f}"
        return f"{x:.4f}"
    return str(x)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"

    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df.iterrows():
        lines.append("| " + " | ".join(format_value(row[c]) for c in cols) + " |")

    return "\n".join(lines)


def write_markdown_report(
    out_path: Path,
    model_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    explain_df: pd.DataFrame,
) -> None:
    lines: list[str] = []

    lines.append("# Evaluation and Improvement Metrics\n")
    lines.append(
        "This report summarizes model, pricing-policy, and explainability improvement metrics.\n"
    )

    lines.append("## 1. Model Improvement Metrics\n")
    lines.append(
        "The naive baseline predicts the mean historical ride price from the training split. "
        "Model improvement is measured against this baseline.\n"
    )
    lines.append(markdown_table(model_df))
    lines.append("")

    lines.append("## 2. Pricing Policy Improvement Metrics\n")
    lines.append(
        "Policy improvements are computed from the counterfactual simulation outputs. "
        "Revenue lift is measured relative to the Historical policy when available.\n"
    )
    lines.append(markdown_table(policy_df))
    lines.append("")

    lines.append("## 3. Explainability Metrics\n")
    lines.append(
        "These metrics summarize how concentrated the SHAP importance distribution is.\n"
    )
    lines.append(markdown_table(explain_df))
    lines.append("")

    lines.append("## Notes\n")
    lines.append(
        "- Simulation metrics are based on a heuristic probabilistic acceptance model, not observed acceptance labels.\n"
        "- SHAP metrics explain model behavior, not causal effects.\n"
        "- v2 metrics are back-transformed to original price scale when target_transform is log.\n"
    )

    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation and improvement metrics for the dynamic pricing project."
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v1", "v2"],
        help="Model artifact versions to evaluate.",
    )
    parser.add_argument(
        "--top-k-shap",
        type=int,
        default=5,
        help="Top-k SHAP features to use for coverage metric.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size for model evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_df = compute_model_improvement_metrics(
        root,
        versions=args.versions,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    policy_df = compute_policy_improvement_metrics(root)
    explain_df = compute_explainability_metrics(root, top_k=args.top_k_shap)

    model_csv = reports_dir / "evaluation_model_improvement.csv"
    policy_csv = reports_dir / "evaluation_policy_improvement.csv"
    explain_csv = reports_dir / "evaluation_explainability_metrics.csv"
    json_path = reports_dir / "evaluation_metrics.json"
    md_path = reports_dir / "evaluation_metrics.md"

    model_df.to_csv(model_csv, index=False)
    policy_df.to_csv(policy_csv, index=False)
    explain_df.to_csv(explain_csv, index=False)

    payload = {
        "model_metrics": model_df.to_dict(orient="records"),
        "policy_metrics": policy_df.to_dict(orient="records"),
        "explainability_metrics": explain_df.to_dict(orient="records"),
    }

    json_path.write_text(json.dumps(payload, indent=2))

    write_markdown_report(md_path, model_df, policy_df, explain_df)

    print(f"Wrote: {model_csv}")
    print(f"Wrote: {policy_csv}")
    print(f"Wrote: {explain_csv}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()