from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse

from dynamic_pricing.features.io import load_csv_file
from dynamic_pricing.models.train import benchmark_models
from dynamic_pricing.models.bundle import ModelBundle
from dynamic_pricing.models.registry import save_model_bundle


def write_report(results, artifact_version, out_path: Path) -> None:
    lines = []
    lines.append(f"# Model Benchmark Report{artifact_version}\n")
    lines.append("Target: `Historical_Cost_of_Ride`\n")
    if artifact_version == "v2":
        lines.append("Transformation: Log")
    else:
        lines.append("Transformation: None")

    lines.append("## Summary Metrics\n")
    lines.append("| Model | MAE | RMSE | R2 | BP p-value (F) | Tuned Params |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for r in results:
        m = r.metrics
        bp_f_p = r.diagnostics.get("breusch_pagan", {}).get("f_pvalue", float("nan"))
        params = r.best_params if r.best_params else {}
        lines.append(
            f"| {r.model_name} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['r2']:.3f} | {bp_f_p:.4g} | {params} |"
        )
    
    lines.append("\n## Diagnostics\n")
    lines.append(f"Figures saved in `reports/figures/{artifact_version}`:\n")
    lines.append("- `qqplot_<model>.png`\n- `resid_vs_fitted_<model>.png`\n- `resid_hist_<model>.png`\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def _format_params(params: dict[str, Any] | None) -> str:
    """
    Markdown-friendly formatting.
    """
    if not params:
        return "-"
    # compact key=val pairs
    items = [f"{k}={v}" for k, v in params.items()]
    return "`" + ", ".join(items) + "`"

def parse_args():
    parser = argparse.ArgumentParser(description="Train and benchmark dynamic pricing models.")
    parser.add_argument(
        "--version",
        type=str,
        choices=["v1", "v2"],
        default="v1",
        help="Model version to train: v1 = price target, v2 = log(price) target",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    artifact_version = args.version

    ROOT = Path(__file__).resolve().parents[1]

    data_path = ROOT / "data" / "raw" / "dynamic_pricing.csv"
    
    artifacts_dir = ROOT / "artifacts" / "model"/ artifact_version
    figures_dir = ROOT / "reports" / "figures" / artifact_version
    reports_path = ROOT / "reports" / f"model_benchmark_{artifact_version}.md"


    print("Running version:", artifact_version)
    print("Saving artifacts to:", artifacts_dir)
    print("Saving figures to:", figures_dir)

    df = load_csv_file(data_path)

    if artifact_version == "v2":
        target_transform = "log"
    else:
        target_transform = "none"

    results, fp = benchmark_models(df, figures_dir=figures_dir, target_transform=target_transform)
    if not results:
        raise RuntimeError("benchmark_models returned no results.")

    best = results[0]
    print("Best model:", best.model_name, best.metrics)

    bundle = ModelBundle(
        model=best.model_obj,
        feature_pipeline=fp,
        model_name=best.model_name,
        model_version=artifact_version,
        feature_version=artifact_version,
    )
    save_model_bundle(bundle, artifacts_dir, target_transform = target_transform)

    write_report(results, artifact_version, reports_path)

    print("Wrote:", reports_path)
    print("Saved bundle to:", artifacts_dir)


if __name__ == "__main__":
    main()