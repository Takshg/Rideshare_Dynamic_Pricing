from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_RAW = [
    "Number_of_Riders", 
    "Number_of_Drivers",
    "Number_of_Past_Rides", 
    "Expected_Ride_Duration",
]

CATEGORICAL = [
    "Location_Category", 
    "Customer_Loyalty_Status",
    "Time_of_Booking", 
    "Vehicle_Type", 
]

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    riders = df["Number_of_Riders"].astype(float)
    drivers = df["Number_of_Drivers"].astype(float).clip(lower=1.0) #avoid div by 0
    duration = df["Expected_Ride_Duration"].astype(float)

    df["riders_per_driver"] = riders / drivers
    df["driver_supply_gap"] = drivers - riders

    df["log_riders"] = np.log1p(riders)
    df["log_drivers"] = np.log1p(df["Number_of_Drivers"].astype(float))
    df["log_duration"] = np.log1p(duration)

    return df

DERIVED_NUMERIC = [
    "riders_per_driver",
    "driver_supply_gap",
    "log_riders",
    "log_drivers",
    "log_duration",
]

@dataclass 
class FeaturePipeline: 
    """"
    Feature pipeline for preprocessing ride-sharing pricing data.

    - fit(df) learns encoders/scalers
    - transform(df) applies transformations to new data
    - feature_names_ provides column order for interpretability and SHAP later

    """

    preprocessor: ColumnTransformer | None = None
    feature_names_: List[str] | None = None
    feature_version: str = "v1.0"

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        df2 = _add_derived_features(df)

        numeric_features = NUMERIC_RAW + DERIVED_NUMERIC
        categorical_features = CATEGORICAL

        missing = [c for c in (numeric_features + categorical_features) if c not in df2.columns]
        if missing:
            raise ValueError(
                f"FeaturePipeline missing columns: {missing}\n"
                f"Available columns: {list(df2.columns)}"
            )

        numeric_pipe = Pipeline(
            steps = [
                ("scaler", StandardScaler()),    
                ]
        )

        categorical_pipe = Pipeline(
            steps = [
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False),)
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric_features), 
                ("cat", categorical_pipe, categorical_features)
            ],
            remainder="drop", 
            verbose_feature_names_out = False
        )

        self.preprocessor.fit(df2)
        self.feature_names_ = self._get_feature_names(numeric_features)
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        if self.preprocessor is None or self.feature_names_ is None:
            raise RuntimeError("FeaturePipline must be fitted before calling transform().")

        df2 = _add_derived_features(df)
        X = self.preprocessor.transform(df2)
        return X, list(self.feature_names_)
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        self.fit(df)
        return self.transform(df)
    
    def _get_feature_names(self, numeric_features: List[str]) -> List[str]:
        assert self.preprocessor is not None
        # Numeric feature names are unchanged, but categorical names come from OneHotEncoder
        cat_encoder: OneHotEncoder = (
            self.preprocessor.named_transformers_["cat"]
            .named_steps["onehot"]
        )
        cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL))
        return list(numeric_features) + cat_feature_names