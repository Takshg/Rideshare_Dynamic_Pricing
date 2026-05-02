from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from dynamic_pricing.api.deps import get_model_bundle
from dynamic_pricing.api.explain import explain_linear_prediction
from dynamic_pricing.schemas.api import (
    ExplainPredictionResponse,
    PricePredictionRequest,
    PricePredictionResponse,
)


router = APIRouter()


@router.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@router.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post("/predict", response_model=PricePredictionResponse)
def predict_price(payload: PricePredictionRequest) -> PricePredictionResponse:
    bundle = get_model_bundle()

    df = pd.DataFrame([payload.model_dump()])

    X, _ = bundle.feature_pipeline.transform(df)
    prediction = float(bundle.model.predict(X)[0])

    return PricePredictionResponse(
        predicted_price=round(prediction, 2),
        model_name=bundle.model_name,
        model_version=bundle.model_version,
        feature_version=bundle.feature_version,
    )


@router.post("/explain", response_model=ExplainPredictionResponse)
def explain_price(payload: PricePredictionRequest) -> ExplainPredictionResponse:
    bundle = get_model_bundle()
    df = pd.DataFrame([payload.model_dump()])

    try:
        result = explain_linear_prediction(bundle, df, top_n=10)
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ExplainPredictionResponse(
        predicted_price=round(result["predicted_price"], 2),
        model_name=bundle.model_name,
        model_version=bundle.model_version,
        feature_version=bundle.feature_version,
        top_contributions=result["top_contributions"],
        explanation_note=result["explanation_note"],
    )