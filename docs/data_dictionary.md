# Data Dictionary - Rideshare Dynamic Pricing

This document defines the raw dataset schema used by the dynamic pricing system.

All column names are **case-sensitive** and must match exactly.

The dataset file must be located at: `data/raw/dynamic_pricing.csv`

## Required Columns

### 1. Number_of_Riders

- Type: Integer
- Description: Number of active rider requests in the area at booking time.
- Range: ≥ 0
- Used for: Demand estimation, derived features, surge modeling.

### 2. Number_of_Drivers

- Type: Integer
- Description: Number of available drivers in the area at booking time.
- Range: ≥ 0
- Used for: Supply estimation, derived features.

---

### 3. Location_Category

- Type: Categorical (string)
- Allowed Values:
  - Urban
  - Suburban
  - Rural
- Description: Geographic classification of booking location.
- Used for: Demand/supply context modeling.

---

### 4. Customer_Loyalty_Status

- Type: Categorical (string)
- Allowed Values (typical):
  - Regular
  - Silver
  - Gold
  - Platinum
- Description: Rider loyalty tier.
- Used for: Pricing fairness analysis, segmentation.

---

### 5. Number_of_Past_Rides

- Type: Integer
- Description: Total completed rides for the customer.
- Range: ≥ 0
- Used for: Experience proxy, segmentation.

---

### 6. Average_Ratings

- Type: Float
- Range: 0.0 – 5.0
- Description: Rider's average rating.
- Used for: Behavioral proxy and segmentation.

---

### 7. Time_of_Booking

- Type: Categorical (string)
- Allowed Values (example):
  - Morning
  - Afternoon
  - Evening
  - Night
- Description: Time-of-day bucket for booking.
- Used for: Demand seasonality.

---

### 8. Vehicle_Type

- Type: Categorical (string)
- Allowed Values:
  - Economy
  - Premium
- Description: Requested ride tier.
- Used for: Base pricing segmentation.

---

### 9. Expected_Ride_Duration

- Type: Integer
- Unit: Minutes
- Range: ≥ 0
- Description: Estimated trip duration.
- Used for: Core pricing signal.

---

### 10. Historical_Cost_of_Ride

- Type: Float
- Unit: Currency (e.g., USD)
- Range: ≥ 0
- Description: Historical ride price under legacy pricing model.
- Used as:
  - Baseline prediction target (Phase 3)
  - Reference for policy comparisons

---

# Data Validation Rules

The CSV loader enforces:

- Exact column name matching
- Header whitespace stripping
- Numeric type coercion
- Categorical whitespace trimming
- Error raised for missing or unexpected columns

---

# Important Notes

- The system assumes no missing values after loader normalization.
- Any schema modification requires updating:
  - `EXPECTED_COLUMNS`
  - Feature pipeline logic
  - Version bump in feature pipeline metadata
