from __future__ import annotations

import pandas as pd

from dynamic_pricing.simulation.acceptance import (
    acceptance_probability,
    expected_revenue,
)


def run_simulation(
    df: pd.DataFrame,
    policy,
    *,
    price_sensitivity: float = 4.0,
) -> pd.DataFrame:
    results = []

    for _, row in df.iterrows():
        proposed_price = policy.price(row)
        base_price = float(row["Historical_Cost_of_Ride"])

        p_accept = acceptance_probability(
            proposed_price=proposed_price,
            base_price=base_price,
            price_sensitivity=price_sensitivity,
        )

        exp_revenue = expected_revenue(
            proposed_price=proposed_price,
            base_price=base_price,
            price_sensitivity=price_sensitivity,
        )

        results.append(
            {
                "policy": policy.name,
                "base_price": base_price,
                "proposed_price": proposed_price,
                "price_delta": proposed_price - base_price,
                "price_delta_pct": (proposed_price - base_price) / base_price,
                "acceptance_probability": p_accept,
                "expected_revenue": exp_revenue,
            }
        )

    return pd.DataFrame(results)


def summarize(results: pd.DataFrame) -> dict:
    return {
        "avg_price": results["proposed_price"].mean(),
        "avg_price_delta_pct": results["price_delta_pct"].mean(),
        "avg_acceptance_probability": results["acceptance_probability"].mean(),
        "total_expected_revenue": results["expected_revenue"].sum(),
        "avg_expected_revenue": results["expected_revenue"].mean(),
        "price_std": results["proposed_price"].std(),
        "max_price": results["proposed_price"].max(),
        "min_price": results["proposed_price"].min(),
    }