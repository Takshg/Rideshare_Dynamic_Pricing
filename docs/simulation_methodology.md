# Simulation Methodology

## Overview

This phase introduces a counterfactual simulation framework to evaluate pricing strategies under a controlled environment. The goal is to move beyond predictive modeling and assess how different pricing policies perform in terms of expected revenue and customer acceptance.

Due to the absence of real demand-response or acceptance data in the dataset, this simulation uses a probabilistic model to approximate customer behavior.

---

## Problem Setting

The dataset provides:

- Historical ride prices (`Historical_Cost_of_Ride`)
- Contextual features (riders, drivers, location, time, etc.)

However, it does NOT include:

- Whether a customer accepted or rejected a ride
- Demand at different price levels
- Elasticity or willingness-to-pay signals

Therefore, we cannot directly model true demand. Instead, we simulate customer response.

---

## Simulation Architecture

The simulation pipeline follows: Context → Pricing Policy → Proposed Price → Acceptance Model → Expected Revenue


### Components

#### 1. Context

Each row in the dataset represents a ride scenario with features such as:

- Number of riders/drivers
- Location category
- Time of booking
- Vehicle type

#### 2. Pricing Policy

A policy is a function that maps a ride context to a proposed price.

Implemented policies include:

- **HistoricalPolicy**

  - Returns the original historical price
  - Serves as baseline
- **RidgeModelPolicy**

  - Uses trained Ridge regression model (v1)
  - Predicts price based on engineered features
- **DemandSupplyPolicy**

  - Rule-based multiplier using rider/driver ratio
  - Simulates surge pricing behavior

---

## Acceptance Model

Since acceptance data is not available, we introduce a **probabilistic acceptance function**.

### Functional Form

\[
P(\text{accept}) = \frac{1}{1 + e^{k \cdot \Delta}}
\]

Where:

- \( \Delta = \frac{\text{proposed_price} - \text{base_price}}{\text{base_price}} \)
- \( k \) = price sensitivity parameter

### Interpretation

- If proposed price ≈ base price → moderate/high acceptance
- If proposed price > base price → acceptance decreases
- If proposed price < base price → acceptance increases

### Implementation Details

- Logistic function ensures smooth, bounded probabilities
- Acceptance probability is clipped to avoid extreme values:
  - min ≈ 0.05
  - max ≈ 0.98

---

## Revenue Model

Instead of assuming fixed demand, revenue is modeled as:

\[
\text{Expected Revenue} = \text{Proposed Price} \times P(\text{accept})
\]

This captures the tradeoff:

- Higher prices increase revenue per ride
- But decrease likelihood of acceptance

---

## Evaluation Metrics

Each policy is evaluated using aggregated statistics:

| Metric                     | Description                      |
| -------------------------- | -------------------------------- |
| avg_price                  | Average proposed price           |
| avg_price_delta_pct        | Avg % change vs historical price |
| avg_acceptance_probability | Mean acceptance likelihood       |
| total_expected_revenue     | Total simulated revenue          |
| avg_expected_revenue       | Per-ride expected revenue        |
| price_std                  | Price volatility                 |
| max_price                  | Maximum surge level              |
| min_price                  | Minimum price                    |

---

## Outputs

Simulation produces:

### 1. Summary Report: 

reports/simulation_results.md

Contains aggregated metrics per policy.

### 2. Detailed Results

reports/simulation_detail_`<policy>`.csv


Contains per-ride:

- proposed price
- price change
- acceptance probability
- expected revenue

---

## Key Assumptions

1. **Historical price approximates baseline willingness-to-pay**
2. **Customer response depends only on relative price change**
3. **Acceptance follows a logistic function**
4. **No supply constraints or system capacity limits are modeled**
5. **Each ride is independent**

---

## Limitations

This simulation is **not causal** and should not be interpreted as real-world revenue prediction.

Limitations include:

- No true demand or elasticity data
- Acceptance model is heuristic, not learned
- Does not account for:
  - competition
  - driver availability constraints
  - repeated customer behavior
  - long-term effects of pricing

---

## Design Rationale

Despite limitations, this approach provides:

- A consistent framework for comparing policies
- A realistic tradeoff between price and acceptance
- A bridge between prediction and decision-making
- A foundation for future improvements

---

## Future Improvements

The simulation can be improved by:

1. Learning acceptance model from real data (if available)
2. Introducing elasticity estimation (revisiting Phase 2)
3. Modeling supply-side constraints
4. Adding dynamic or sequential decision-making
5. Implementing reinforcement learning for pricing policies

---

## Conclusion

This simulation transforms the project from a predictive model into a pricing decision system. While simplified, it provides meaningful insights into how different pricing strategies behave under uncertainty and establishes a strong foundation for more advanced demand modeling.
