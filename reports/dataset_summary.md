# AI4I 2020 Dataset Summary

## Dataset Profile
- Source file: `data/raw/ai4i2020.csv`
- Scale: 10,000 rows x 14 original columns.
- Target: `Machine failure`, with 339 positive samples and 9,661 normal samples.
- Class imbalance: failure rate is 3.39%.
- Quality categories: `L`, `M`, `H`; physical sensor fields include temperature, rotational speed, torque, and tool wear.
- Missing values: see `reports/missing_values.csv`; this dataset has no missing values in the original AI4I file if all counts are zero.

## Cleaning And Standardization
- Preserved identifiers (`UDI`, `Product ID`) for traceability.
- Added `failure_mode`, `temperature_gap [K]`, and `power_proxy`.
- Standardized all numeric sensor features into z-score columns.
- Added one-hot columns for product type.

## Federated Factory Splits
- IID: stratified by product type and machine failure label. Factory row range: 1998-2002.
- Moderate Non-IID: biased by product type and tool wear while preserving both target classes across factories. Factory row range: 1940-2042.
- Highly Non-IID: concentrates failure modes and normal samples into dominant factories to create strong client heterogeneity. Factory row range: 1957-2032.

## Presentation Figures
- `figures/class_distribution.png`
- `figures/type_distribution.png`
- `figures/failure_mode_distribution.png`
- `figures/correlation_heatmap.png`
- `figures/client_distribution_iid.png`
- `figures/client_distribution_moderate_non_iid.png`
- `figures/client_distribution_highly_non_iid.png`
