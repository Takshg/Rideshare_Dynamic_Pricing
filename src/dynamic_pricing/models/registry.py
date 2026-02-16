from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone 
from pathlib import Path
import json
import joblib

from dynamic_pricing.models.bundle import ModelBundle
from dynamic_pricing.features.pipeline import FeaturePipeline

def save_feature_pipeline(
    fp: FeaturePipeline, 
    out_dir: str | Path,
    *, 
    training_rows: int, 
    training_cols: int, 
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save pipeline object (includes fitted sklearn objects)
    joblib.dump(fp, out_dir/"feature_pipeline.joblib")

    metadata = {
        "type": "feature_pipeline",
        "feature_version": fp.feature_version, 
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_shape": [training_rows, training_cols],   
    }
    
    (out_dir/"metadata.json").write_text(json.dumps(metadata, indent=2))

def load_feature_pipeline(in_dir: str | Path) -> FeaturePipeline:
    in_dir = Path(in_dir) 
    obj = joblib.load(in_dir/"feature_pipeline.joblib")
    if not isinstance(obj, FeaturePipeline):
        raise TypeError("Loaded object is not a FeaturePipeline") 
    return obj

def save_model_bundle(bundle: ModelBundle, out_dir: str | Path, *,  target_transform: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, out_dir / "model_bundle.joblib")

    meta = {
        "artifact_type": "model_bundle",
        "model_name": bundle.model_name,
        "model_version": bundle.model_version,
        "feature_version": bundle.feature_version,
        "target_transform": target_transform,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

def load_model_bundle(in_dir: str | Path) -> ModelBundle:
    in_dir = Path(in_dir)
    return joblib.load(in_dir / "model_bundle.joblib")