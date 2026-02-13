from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelBundle:
    model: Any
    feature_pipeline: Any
    model_name: str
    model_version: str
    feature_version: str