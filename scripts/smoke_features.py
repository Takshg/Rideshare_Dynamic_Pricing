### Smoke Test Script for Feature Pipeline (Fit -> Transform -> Save -> Reload)

from __future__ import annotations 
from dynamic_pricing.features.io import load_csv_file
from dynamic_pricing.features.pipeline import FeaturePipeline
from dynamic_pricing.models.registry import save_feature_pipeline, load_feature_pipeline

def main() -> None: 
    df = load_csv_file("data/raw/dynamic_pricing.csv")
    print("Loaded data shape:", df.shape)

    fp = FeaturePipeline()
    X, names = fp.fit_transform(df)

    print("X shape:", X.shape)
    print("Num features:" , len(names))
    print("First 15 feature names:", names[:15])

    if X.shape[0] != df.shape[0]:
        raise RuntimeError("Row count mismatch between X and df")
    if X.shape[1] != len(names):
        raise RuntimeError("Feature name count mismatch")
    if (X != X).any(): 
        raise RuntimeError("NaN values found in feature matrix X")
    
    save_feature_pipeline(
        fp, 
        "artifacts/feature_pipeline_smoke_test_result", 
        training_rows = df.shape[0], 
        training_cols = df.shape[1],
    )

    fp2 = load_feature_pipeline("artifacts/feature_pipeline_smoke_test_result")
    X2, names2 = fp2.transform(df)

    assert X2.shape == X.shape 
    assert names2 == names
    print("Reload check: OK")

if __name__ == "__main__": 
    main()
    