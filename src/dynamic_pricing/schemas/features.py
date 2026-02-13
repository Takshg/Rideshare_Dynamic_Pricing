from __future__  import annotations
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, conint, confloat

class LocationCategory(str, Enum):
    URBAN = "urban"
    SUBURBAN = "suburban"
    RURAL = "rural"

class LoyaltyStatus(str, Enum):
    REGULAR = "regular"
    SILVER = "silver"
    GOLD = "gold"

class VehicleType(str, Enum):
    ECONOMY = "economy"
    PREMIUM = "premium"

class TimeOfBooking(str, Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"

class RideContext(BaseModel):
    """
    Conical schema for a single ride context
    """
    
    number_of_riders: conint(ge=0, le=500) = Field(..., description="Number_of_Riders")
    number_of_drivers: conint(ge=0, le=500) = Field(..., description="Number_of_Drivers")

    location_category: LocationCategory = Field(..., description="Location_Category")
    customer_loyalty_status: LoyaltyStatus = Field(..., description="Customer_Loyalty_Status")

    number_of_past_rides: conint(ge=0, le=10000) = Field(..., description="Number_of_Past_Rides")
    average_rating: confloat(ge=1.0, le=5.0) = Field(..., description="Average_Rating")

    time_of_booking: TimeOfBooking = Field(..., description="Time_of_Booking")
    vehicle_type: VehicleType = Field(..., description="Vehicle_Type")

    expected_ride_duration: conint(ge=0, le = 10000) = Field(..., description="Expected_Ride_Duration")

    historical_cost_of_ride: Optional[confloat(ge=0.0)] = Field(default=None, description="Historical_Cost_of_Ride")

    class Config:
        populate_by_name = True