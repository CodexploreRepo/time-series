# EDA - Stationary & Differentiation

Before fitting an ARIMA model to time series data, it is important to conduct an exploratory analysis to determine the following:
- **Stationarity**: Stationarity means that the statistical properties (mean, variance...) remain constant over time, so time series with trends or seasonality are not stationary. Since ARIMA assumes the stationarity of the data, it is essential to subject the data to rigorous tests, such as the Augmented Dickey-Fuller test, to assess stationarity. 
    - If *non-stationarity* is found, the series should be differenced until stationarity is achieved. This analysis helps to determine the optimal value of the parameter  $𝑑$.

- **Autocorrelation analysis**: Plot the autocorrelation and partial autocorrelation functions (ACF and PACF) to identify potential lag relationships between data points. This visual analysis provides insight into determining appropriate autoregressive (AR) and moving average (MA) terms ($𝑝$ and  $𝑞$) for the ARIMA model.

- **Seasonal decomposition**: In cases where seasonality is suspected, decomposing the series into trend, seasonal, and residual components using techniques such as moving averages or `seasonal time series decomposition` (STL) can reveal hidden patterns and help identify seasonality. This analysis helps to determine the optimal values of the parameters  $𝑃$, $𝐷$, $𝑄$ and  $𝑚$.

## Stationary
### Augmented Dickey-Fuller test
- **Augmented Dickey-Fuller** test takes as its null hypothesis that the time series has a **unit root** - a characteristic of non-stationary time series. Conversely, the alternative hypothesis (under which the null hypothesis is rejected) is that the series is stationary.
    - Null Hypothesis ($H_0$): The series is not stationary or has a unit root.
    - Alternative hypothesis ($H_A$): The series is stationary with no unit root.
- **Rule**: the p-value obtained should be less than a specified significance level, often set at 0.05, to reject this hypothesis.
```Python
from statsmodels.tsa.stattools import adfuller

print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
```
### Kwiatkowski-Phillips-Schmidt-Shin test (KPSS)

- The KPSS test checks if a time series is stationary around a mean or linear trend. 
- **Rule**: small p-values (e.g., less than 0.05) rejects the null hypothesis and suggest that differencing is required. 

```Python
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')
```

