# Autocorrelation Analysis

- **Autocorrelation Function (ACF)** helps in identifying the value of $q$ (lag in the moving average part)
- **Partial Autocorrelation Function (PACF)** assists in identifying the value of $p$ (lag in the autoregressive part).
- Conclusion: The difference between ACF and PACF is the inclusion or exclusion of indirect correlations in the calculation.
  - Note: If the **stationarity analysis** indicates that **differencing is required**, **subsequent analysis** should be conducted **using the differenced series**

## Intuition

- Let $S_t$ is the average price of salmon this month
- $S_t$ or the average price of salmon this month be affected by $S_{t-1}$
- $S_t$ is also affected by the price 2 months ago $S_{t-2}$ in two different ways:
  - Directly: $S_{t-2}$ &#8594; $S_t$
  - In-directly: $S_{t-2}$ &#8594; $S_{t-1}$ &#8594; $S_t$
  - Hence, the correlation $corr(S_{t-2}, S_t)$ which captures both **direct & in-direct** impact of the price 2 months ago, $S_{t-2}$, on the current price $S_t$, is the `ACF`
- PACF only focus on the direct impact $S_{t-2}$ &#8594; $S_t$

## Autocorrelation Function (ACF)

- The autocorrelation function (ACF) measures the linear relationship between lagged values of a time series.
  - In other words, it measures the correlation of the time series with itself.
  - The autocorrelation coefficient between $y_t$ and its lag=0 or $y_{t-0}=y_t$ is always $1$
  - The autocorrelation coefficient between $y_t$ and its lag=1 or $y_{t-1}$ is $r_1$
  - The autocorrelation coefficient between $y_t$ and its lag=2 or $y_{t-2}$ is $r_2$
- Understanding the ACF plot:
  - In the presence of a trend, a plot of the ACF will show that the coefficients are high for short lags, and they will decrease linearly as the lag increases.
  - In the presence of a seasonal, the ACF plot will also display cyclical patterns.
    - Note: If the ACF displays a sinusoidal or damped sinusoidal pattern, it suggests seasonality is present and requires **consideration of seasonal orders** in addition to non-seasonal orders.
  - The **shaded area** represents a confidence interval.
    - If a point is within the shaded area, then it is not significant.
    - Otherwise, the autocorrelation coefficient is significant.

## Partial Autocorrelation Function (PACF)

- The PACF measures the correlation between a lagged value and the current value of the time series, while accounting for the effect of the intermediate lags.
- In the context of ARIMA modeling, if the PACF sharply cuts off after a certain lag, while the remaining values are within the confidence interval, it suggests an AR model of that order.
- The lag, at which the PACF cuts off, gives an idea of the value of $p$.
