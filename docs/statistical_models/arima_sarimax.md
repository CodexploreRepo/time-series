# ARIMA and SARIMAX

## Introduction

- Statistical forecasting models:
  - `ARIMA` (AutoRegressive Integrated Moving Average): AR (`𝑝`), I `d`, MA (`q`)
  - `SARIMAX` (**Seasonal** AutoRegressive Integrated Moving Average with **eXogenous** regressors): ARIMA framework + seasonal patterns + exogenous variables
- This model comprises three components.
  - **The autoregressive element (AR)** assumes that the current value ($y_t$) is dependent on previous values ($y_{t-1}$, $y_{t-2}$, …). Because of this assumption, we can build a linear regression model.
    - To figure out the order `𝑝` of an AR model, you need to look at the PACF.
  - **The moving average element (MA)** assumes that the regression error is a linear combination of past forecast errors.
    - To figure out the order `q` of an MA model, you need to look at the ACF.
  - **The integrated component (I)** indicates that the data values have been replaced with the difference between their values and the previous ones (and this differencing process may have been performed more than once).
- In the ARIMA-SARIMAX model notation, the parameters
  - `𝑝`, `𝑑` and `𝑞` represent the autoregressive, differencing, and moving-average components
    - `𝑝` is the order (number of time lags) of the autoregressive part of the model.
    - `𝑑` is the degree of differencing (the number of times that past values have been subtracted from the data).
    - `𝑞` is the order of the moving average part of the model.
  - `𝑃`, `𝐷` and `𝑄` denote the same components for the _seasonal_ part of the model, with `𝑚` representing the number of periods in each season.

### Python-Library for ARIMA and SARIMAX

- Several Python libraries implement ARIMA-SARIMAX models. Four of them are:
  - `statsmodels`: is one of the most complete libraries for statistical modeling in Python. Its API is often more intuitive for those coming from the R environment than for those used to the object-oriented API of scikit-learn.
  - `pmdarima`: This is a wrapper for `statsmodels` SARIMAX. Its distinguishing feature is its seamless integration with the scikit-learn API, allowing users familiar with `scikit-learn`'s conventions to seamlessly dive into time series modeling.
  - `skforecast`: Among its many forecasting features, it has a new wrapper of `statsmodels` SARIMAX that also follows the scikit-learn API. This implementation is very similar to that of `pmdarima`, but has been simplified to include only the essential elements for skforecast, resulting in significant speed improvements.
  - `statsForecast`: It offers a collection of widely used univariate time series forecasting models, including automatic ARIMA, ETS, CES, and Theta modeling optimized for high performance using `numba`.

## ARIMA & SARIMAX Modeling

### EDA

Before fitting an ARIMA model to time series data, it is important to conduct an exploratory analysis to determine the following:

- **Stationarity**: Stationarity means that the statistical properties (mean, variance...) remain constant over time, so time series with trends or seasonality are not stationary. Since ARIMA assumes the stationarity of the data, it is essential to subject the data to rigorous tests, such as the Augmented Dickey-Fuller test, to assess stationarity.

  - If _non-stationarity_ is found, the series should be differenced until stationarity is achieved. This analysis helps to determine the optimal value of the parameter $𝑑$.

- **Autocorrelation analysis**: Plot the autocorrelation and partial autocorrelation functions (ACF and PACF) to identify potential lag relationships between data points. This visual analysis provides insight into determining appropriate autoregressive (AR) and moving average (MA) terms ($𝑝$ and $𝑞$) for the ARIMA model.

- **Seasonal decomposition**: In cases where seasonality is suspected, decomposing the series into trend, seasonal, and residual components using techniques such as moving averages or `seasonal time series decomposition` (STL) can reveal hidden patterns and help identify seasonality. This analysis helps to determine the optimal values of the parameters $𝑃$, $𝐷$, $𝑄$ and $𝑚$.

#### Stationary

- A stationary time series is one whose statistical properties do not change over time
  - Namely: constant mean, variance, and autocorrelation, and these properties are independent of time.
- Intuitively, this makes sense, because if the data is non-stationary, its properties are going to change over time, which would mean that our model parameters must also change through time.
- The function `check_stationarity` combines both ADF and KPSS tests for checking the stationary in the time series.

```Python
from statsmodels.tsa.stattools import adfuller, kpss

def check_stationarity(series, p_significant=0.05):
    # Copied from https://machinelearningmastery.com/time-series-data-stationary-python/

    adfuller_result = adfuller(series.values)
    kpss_result = kpss(series.values)

    print(f'ADF Statistic : {adfuller_result[0]:.5f}, p-value: {adfuller_result[1]:.5f}')
    print('Critical Values:')
    for key, value in adfuller_result[4].items():
        print('\t%s: %.3f' % (key, value))
    print(f'KPSS Statistic: {kpss_result[0]:.5f}, p-value: {kpss_result[1]:.5f}')
    if (adfuller_result[1] <= p_significant) & (adfuller_result[4]['5%'] > adfuller_result[0]) & (kpss_result[1] > p_significant):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")
```

#### Autocorrelation Analysis

- Once a process is stationary, plotting the autocorrelation function (ACF) & Partial Autocorrelation Function (PACF) is a great way to understand what type of process you are analyzing.

##### Autocorrelation Function (ACF)

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

##### Partial Autocorrelation Function (PACF)

- The PACF measures the correlation between a lagged value and the current value of the time series, while accounting for the effect of the intermediate lags.
- In the context of ARIMA modeling, if the PACF sharply cuts off after a certain lag, while the remaining values are within the confidence interval, it suggests an AR model of that order.
- The lag, at which the PACF cuts off, gives an idea of the value of $p$.

##### How to determine Order of AR, MA, and ARMA Models with ACF & PACF

- Examples: Tails off (Geometric decay) & Cuts off after lag `k` patterns

|              Tails off               |              Cuts off               |
| :----------------------------------: | :---------------------------------: |
| ![](../assets/img/acf-tail-off.webp) | ![](../assets/img/pacf-cutoff.webp) |

- To determine the order of the model, you can use the following table:

|      | AR($p, q=0$)                                         | MA($p=0, q$)                                    | ARMA($p$, $q$)              |
| ---- | ---------------------------------------------------- | ----------------------------------------------- | --------------------------- |
| ACF  | Tails off (Geometric decay)                          | Significant at lag $q$ / Cuts off after lag $q$ | Tails off (Geometric decay) |
| PACF | Significant at each lag $p$ / Cuts off after lag $p$ | Tails off (Geometric decay)                     | Tails off (Geometric decay) |

##### Rules of Thumb

- If only the PACF cuts off after lag p (ACF tails off), one could start with an $AR(p)$ model.
- If only the ACF cuts off after lag q (PACF tails off), one could start with an $MA(q)$ model.
- For real life times series dataset, both ACF and PACF plots are not clear whether they are tailing off or cutting off, so AR & MA models are not practically used.
  - In fact, the model ARMA & ARIMA (with differencing) $ARIMA(p, d, q)$ are used more frequently by determine the combination of $p$, $q$ to get the better score of AIC and BIC.
