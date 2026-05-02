from __future__ import annotations

from pydantic import BaseModel, Field


class PricePredictionRequest(BaseModel):
    Number_of_Riders: int = Field(..., ge=0)
    Number_of_Drivers: int = Field(..., ge=0)
    Location_Category: str
    Customer_Loyalty_Status: str
    Number_of_Past_Rides: int = Field(..., ge=0)
    Average_Ratings: float = Field(..., ge=0, le=5)
    Time_of_Booking: str
    Vehicle_Type: str
    Expected_Ride_Duration: int = Field(..., ge=0)
    Historical_Cost_of_Ride: float | None = Field(default=None, ge=0)


class PricePredictionResponse(BaseModel):
    predicted_price: float
    model_name: str
    model_version: str
    feature_version: str