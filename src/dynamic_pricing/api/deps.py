from __future__ import annotations

from pathlib import Path

from dynamic_pricing.models.registry import load_model_bundle


ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "artifacts" / "model" / "v1"


_model_bundle = None


def get_model_bundle():
    global _model_bundle

    if _model_bundle is None:
        _model_bundle = load_model_bundle(MODEL_DIR)

    return _model_bundle