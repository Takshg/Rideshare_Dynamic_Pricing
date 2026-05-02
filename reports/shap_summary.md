# SHAP Explainability Summary

Model artifact: `artifacts/model/v1/`

Feature pipeline: saved inside model bundle

## Generated Outputs

- Summary beeswarm: `/Users/takshgirdhar/Desktop/Projects/Rideshare_Dynamic_Pricing/reports/figures/shap/shap_summary_beeswarm.png`
- Feature importance bar plot: `/Users/takshgirdhar/Desktop/Projects/Rideshare_Dynamic_Pricing/reports/figures/shap/shap_feature_importance_bar.png`
- Importance CSV: `/Users/takshgirdhar/Desktop/Projects/Rideshare_Dynamic_Pricing/reports/figures/shap/shap_feature_importance.csv`

## Top Features by Mean Absolute SHAP

| Rank | Feature | Mean Absolute SHAP |
|---:|---|---:|
| 1 | `Expected_Ride_Duration` | 162.4440 |
| 2 | `Vehicle_Type_Economy` | 10.5429 |
| 3 | `Vehicle_Type_Premium` | 10.5429 |
| 4 | `log_riders` | 7.2136 |
| 5 | `log_duration` | 6.3842 |
| 6 | `Number_of_Riders` | 5.6030 |
| 7 | `Number_of_Drivers` | 4.0953 |
| 8 | `riders_per_driver` | 2.8494 |
| 9 | `log_drivers` | 2.7482 |
| 10 | `driver_supply_gap` | 2.4908 |
| 11 | `Location_Category_Rural` | 1.3330 |
| 12 | `Customer_Loyalty_Status_Gold` | 1.2966 |
| 13 | `Time_of_Booking_Evening` | 1.2660 |
| 14 | `Customer_Loyalty_Status_Silver` | 1.0722 |
| 15 | `Location_Category_Urban` | 0.9562 |