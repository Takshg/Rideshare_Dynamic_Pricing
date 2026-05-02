from __future__ import annotations

import pandas as pd
from fastapi import APIRouter

from dynamic_pricing.api.deps import get_model_bundle
from dynamic_pricing.schemas.api import PricePredictionRequest, PricePredictionResponse


router = APIRouter()


@router.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@router.post("/predict", response_model=PricePredictionResponse)
def predict_price(payload: PricePredictionRequest) -> PricePredictionResponse:
    bundle = get_model_bundle()

    row = payload.model_dump()
    df = pd.DataFrame([row])

    X, _ = bundle.feature_pipeline.transform(df)
    prediction = float(bundle.model.predict(X)[0])

    return PricePredictionResponse(
        predicted_price=round(prediction, 2),
        model_name=bundle.model_name,
        model_version=bundle.model_version,
        feature_version=bundle.feature_version,
    )