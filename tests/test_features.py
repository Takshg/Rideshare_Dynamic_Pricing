from __future__ import annotations
import pandas as pd

from dynamic_pricing.features.io import load_csv_file
from dynamic_pricing.features.pipeline import FeaturePipeline
from dynamic_pricing.schemas.features import RideContext

def tes_schema_parses_valid_row() -> None:
    # Mininal realistic row using dataset values
    payload = {
        "Number_of_Riders": 60, 
        "Number_of_Drivers": 25, 
        "Location_Category": "Urban",
        "Customer_Loyalty_Status": "Gold", 
        "Number_of_Past_Rides": 10, 
        "Average_Ratings": 4.2, 
        "Time_of_Booking": "Evening", 
        "Vehicle_Type": "Economy", 
        "Expected_Ride_Duration": 40, 
        "Historical_Cost_of_Ride": 200.0,
    }

    rc = RideContext(**payload)
    assert rc.number_of_riders == 60
    assert rc.vehicle_type.value in {"Economy", "Premium"}

def test_feature_pipeline_fit_transform_stable() -> None: 
    df = load_csv_file("data/raw/dynamic_pricing.csv")
    fp = FeaturePipeline()

    X1, names1 = fp.fit_transform(df)
    X2, names2 = fp.transform(df)
    
    assert X1.shape == X2.shape
    assert names1 == names2
    assert X1.shape[0] == len(df)
    assert X1.shape[1] == len(names1)

def test_feature_pipeline_handles_unknown_categories() -> None:
    df = load_csv_file("data/raw/dynamic_pricing.csv").copy()
    df.loc[0, "Location_Category"] = "NewCityType" #unknown category
    fp = FeaturePipeline()
    fp.fit(df)
    X, names = fp.transform(df)

    assert X.shape[0] == len(df)
    assert X.shape[1] == len(names)
