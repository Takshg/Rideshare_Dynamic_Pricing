from pathlib import Path

import pandas as pd

from dynamic_pricing.features.io import load_csv_file
from dynamic_pricing.models.registry import load_model_bundle
from dynamic_pricing.simulation.policies import (
    HistoricalPolicy,
    RidgeModelPolicy,
    DemandSupplyPolicy,
)
from dynamic_pricing.simulation.simulator import run_simulation, summarize


def main() -> None:
    ROOT = Path(__file__).resolve().parents[1]

    df = load_csv_file(ROOT / "data/raw/dynamic_pricing.csv")
    bundle = load_model_bundle(ROOT / "artifacts/model/v1")

    policies = [
        HistoricalPolicy(),
        RidgeModelPolicy(bundle),
        DemandSupplyPolicy(),
    ]

    summary_results = []

    for policy in policies:
        sim = run_simulation(df, policy, price_sensitivity=4.0)
        summary = summarize(sim)
        summary["policy"] = policy.name
        summary_results.append(summary)

        # Save ride-level simulation output
        out_detail = ROOT / "reports" / f"simulation_detail_{policy.name}.csv"
        sim.to_csv(out_detail, index=False)

    result_df = pd.DataFrame(summary_results)

    print(result_df)

    out_path = ROOT / "reports" / "simulation_results.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_markdown(out_path, index=False)

    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()