from __future__ import annotations

from dynamic_pricing.features.io import load_csv_file
from dynamic_pricing.models.train import benchmark_models

def test_benchmark_runs() -> None:
    df = load_csv_file("data/raw/dynamic_pricing.csv")
    results, fp = benchmark_models(df, test_size=0.2, random_state=42)
    assert len(results) >= 2
    assert results[0].metrics["rmse"] >= 0.0