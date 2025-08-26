# Notes on Model GridSearchCV Parameters

## 1. `max_depth` Recommendation
- **Minimum Value:** `max_depth` should not be set below **20 or 30**.
- **Warning:** Values like `10` may cause crashes in algorithms such as `RandomForestRegressor` or `BaggingRegressor`.

## 2. RandomForestRegressor Specifics
- **Metrics:** Avoid using `"absolute_error"` in the grid search metrics, as it may raise errors and crash the algorithms due to **NaN values** in lag features.
- **Overfitting Risk:** Setting `ccp_alpha = 1` can lead to overfitting.
