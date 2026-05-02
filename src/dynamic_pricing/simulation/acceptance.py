from __future__ import annotations

import numpy as np
import pandas as pd


def acceptance_probability(
    proposed_price: float,
    base_price: float,
    *,
    price_sensitivity: float = 4.0,
    min_accept_prob: float = 0.05,
    max_accept_prob: float = 0.98,
) -> float:
    """
    Simulates probability of customer acceptance.

    Assumption:
    - If proposed_price == base_price, acceptance is moderate-high.
    - If proposed_price > base_price, acceptance decreases.
    - If proposed_price < base_price, acceptance increases.

    This is a simulation assumption, not learned from true acceptance data.
    """
    base_price = max(float(base_price), 1e-8)
    relative_price_change = (proposed_price - base_price) / base_price

    # Higher price increase => lower probability
    prob = 1 / (1 + np.exp(price_sensitivity * relative_price_change))

    return float(np.clip(prob, min_accept_prob, max_accept_prob))


def expected_revenue(
    proposed_price: float,
    base_price: float,
    *,
    price_sensitivity: float = 4.0,
) -> float:
    """
    Expected revenue = price * probability of acceptance.
    """
    p_accept = acceptance_probability(
        proposed_price,
        base_price,
        price_sensitivity=price_sensitivity,
    )
    return float(proposed_price * p_accept)