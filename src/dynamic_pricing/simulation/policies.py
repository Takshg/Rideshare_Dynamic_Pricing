from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

class PricingPolicy:
    name : str

    def price(self, row: pd.Series) -> float:
        raise NotImplementedError
    
@dataclass
class HistoricalPolicy(PricingPolicy):
    name: str = "Historical"
    
    def price(self, row:pd.Series) -> float: 
        return float(row["Historical_Cost_of_Ride"])

@dataclass

@dataclass
class RidgeModelPolicy(PricingPolicy):
    model_bundle: any
    name: str = "ridge_model"

    def price(self, row: pd.Series) -> float:
        fp = self.model_bundle.feature_pipeline
        model = self.model_bundle.model

        df = pd.DataFrame([row])
        X, _ = fp.transform(df)

        pred = model.predict(X)[0]
        return float(pred)
    
@dataclass
class DemandSupplyPolicy(PricingPolicy):
    base_multiplier: float = 1.0
    max_multiplier: float = 2.0
    min_multiplier: float = 0.5
    name: str = "demand_supply"

    def price(self, row: pd.Series) -> float:
        riders = float(row["Number_of_Riders"])
        drivers = max(float(row["Number_of_Drivers"]), 1.0)

        ratio = riders / drivers

        multiplier = 1 + 0.5 * (ratio - 1)
        multiplier = max(self.min_multiplier, min(self.max_multiplier, multiplier))

        base_price = float(row["Historical_Cost_of_Ride"])
        return base_price * multiplier