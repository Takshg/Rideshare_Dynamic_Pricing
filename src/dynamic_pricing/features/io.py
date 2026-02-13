from __future__ import annotations
from pathlib import Path 
import pandas as pd

# Fallback column names for input data
EXPECTED_COLUMNS = [
    "Number_of_Riders", 
    "Number_of_Drivers",
    "Location_Category",
    "Customer_Loyalty_Status",
    "Number_of_Past_Rides", 
    "Average_Ratings",
    "Time_of_Booking",
    "Vehicle_Type",
    "Expected_Ride_Duration",
    "Historical_Cost_of_Ride", 
] 

def load_csv_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip()
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    if missing: 
        raise ValueError(f"Input data is missing expected columns: {missing}")
    if extra: 
        raise ValueError(f"Input data has unexpected extra columns: {extra}")
    
    # Data type normalization
    numeric_cols = [
        "Number_of_Riders", 
        "Number_of_Drivers",
        "Number_of_Past_Rides", 
        "Average_Ratings", 
        "Expected_Ride_Duration",
        "Historical_Cost_of_Ride",    
    ]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Strip whitespace from categorical columns
    cat_cols = [
        "Location_Category",  
        "Customer_Loyalty_Status", 
        "Time_of_Booking",
        "Vehicle_Type",
    ]
    
    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip()
    
    return df