| avg_price | avg_price_delta_pct | avg_acceptance_probability | total_expected_revenue | avg_expected_revenue | price_std | max_price | min_price | policy        |
| --------: | ------------------: | -------------------------: | ---------------------: | -------------------: | --------: | --------: | --------: | :------------ |
|   372.503 |                   0 |                        0.5 |                 186251 |              186.251 |   187.159 |   836.116 |   25.9934 | Historical    |
|   373.383 |             0.03256 |                   0.475642 |                 177739 |              177.739 |    175.03 |   685.765 |   29.1624 | ridge_model   |
|   612.915 |            0.647995 |                   0.136175 |                70140.4 |              70.1404 |   330.926 |   1672.23 |   30.6145 | demand_supply |


The probabilistic acceptance model is a simulation assumption because the dataset does not contain real acceptance/rejection outcomes. Acceptance probability is modeled as a decreasing logistic function of relative price increase compared with historical price. This allows comparison of pricing policies under a controlled counterfactual framework, but results should not be interpreted as causal demand estimates.
