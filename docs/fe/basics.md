# Feature Engineering - Basics

## Lag Feature

- Lag features are previous values of the time series used as predictors for future values.

```Python
# Create lag features
time_series['lag_1'] = time_series.shift(1) # y(t) = y(t-1)
time_series['lag_2'] = time_series.shift(2) # y(t) = y(t-2)
# Drop NaN values after the lag feature creation
time_series.dropna(inplace=True)
```

## Rolling Window Features

- Rolling window features are statistics (`mean`, `standard deviation`, `minimum`, and `maximum`) computed over a sliding `window` of the past values.
  - **Rolling Mean** is also equivalent to Simple Moving Average (`SMA`) in the technical indicators
  - `window` size depends on the data
    - If the data is collected daily, window size can be 7, 14, 30 days to capture the effects in a week, half a month, or a full month
    - If the data is collected hourly, window size can be 6, 12, 24 hours.

```Python
# Create rolling window features
time_series['rolling_mean_3'] = time_series['value'].rolling(window=3).mean()
time_series['rolling_std_3'] = time_series['value'].rolling(window=3).std()
time_series['rolling_min_3'] = time_series['value'].rolling(window=3).min()
time_series['rolling_max_3'] = time_series['value'].rolling(window=3).max()
# Drop NaN values
time_series.dropna(inplace=True)
```

## Expanding Window Features

- Expanding window features compute statistics (`mean`, `standard deviation`, `minimum`, and `maximum`) over all past values up to the current time point.

```Python
# Create expanding window features
time_series['expanding_mean'] = time_series['value'].expanding().mean()
time_series['expanding_std'] = time_series['value'].expanding().std()
```

## Rate of Change

- The rate of change ($roc_n$) is defined as follows:
  $$roc_n= \frac{y_t - y_{t-n}}{y_{t-n}}$$

```Python
# Create roc features
time_series['roc_period=1'] = time_series['value'].pct_change() # default period=1
time_series['roc_period=5'] = time_series['value'].pct_change(periods=5)
```
