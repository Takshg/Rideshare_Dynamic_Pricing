from __future__ import annotations
from typing import Any, Dict, Optional
from pydantic import BaseModel

class PricePrediction(BaseModel):
    predicted_price: float
    currency: str = "USD"

    model_name: str 
    model_version: str
    feature_version: str

    explanation: Optional[Dict[str, Any]] = None