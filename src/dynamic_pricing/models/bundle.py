from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from dynamic_pricing.features.pipeline import FeaturePipeline

@dataclass
class ModelBundle:
    model: Any
    feature_pipeline: FeaturePipeline
    model_name: str
    model_version: str
    feature_version: str