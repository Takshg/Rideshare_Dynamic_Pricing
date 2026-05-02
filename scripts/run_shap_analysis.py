from __future__ import annotations

from pathlib import Path

from dynamic_pricing.features.io import load_csv_file
from dynamic_pricing.models.registry import load_model_bundle
from dynamic_pricing.explainability.shap_analysis import run_shap_analysis


def write_summary(result: dict, out_path: Path) -> None:
    lines = []
    lines.append("# SHAP Explainability Summary\n")
    lines.append("Model artifact: `artifacts/model/v1/`\n")
    lines.append("Feature pipeline: saved inside model bundle\n")

    lines.append("## Generated Outputs\n")
    lines.append(f"- Summary beeswarm: `{result['summary_plot']}`")
    lines.append(f"- Feature importance bar plot: `{result['bar_plot']}`")
    lines.append(f"- Importance CSV: `{result['importance_csv']}`")

    lines.append("\n## Top Features by Mean Absolute SHAP\n")
    lines.append("| Rank | Feature | Mean Absolute SHAP |")
    lines.append("|---:|---|---:|")

    for i, row in enumerate(result["top_features"], start=1):
        lines.append(
            f"| {i} | `{row['feature']}` | {row['mean_abs_shap']:.4f} |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main() -> None:
    ROOT = Path(__file__).resolve().parents[1]

    df = load_csv_file(ROOT / "data/raw/dynamic_pricing.csv")
    bundle = load_model_bundle(ROOT / "artifacts/model/v1")

    result = run_shap_analysis(
        bundle,
        df,
        output_dir=ROOT / "reports" / "figures" / "shap",
    )

    out_path = ROOT / "reports" / "shap_summary.md"
    write_summary(result, out_path)

    print("SHAP analysis complete.")
    print(f"Summary written to: {out_path}")


if __name__ == "__main__":
    main()