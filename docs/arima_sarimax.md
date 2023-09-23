# ARIMA and SARIMAX

## ARIMA-SARIMAX Introduction 
- Statistical forecasting models:
    - `ARIMA` (AutoRegressive Integrated Moving Average)
    - `SARIMAX` (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors): expand on the ARIMA framework by seamlessly incorporating seasonal patterns and exogenous variables
- This model comprises three components. 
    - **The autoregressive element (AR)** relates the current value to past (lagged) values. 
    - **The moving average element (MA)** assumes that the regression error is a linear combination of past forecast errors. 
    - **The integrated component (I)** indicates that the data values have been replaced with the difference between their values and the previous ones (and this differencing process may have been performed more than once).
- In the ARIMA-SARIMAX model notation, the parameters 
    - `𝑝`, `𝑑` and `𝑞` represent the autoregressive, differencing, and moving-average components
        - `𝑝` is the order (number of time lags) of the autoregressive part of the model.
        - `𝑑` is the degree of differencing (the number of times that past values have been subtracted from the data).
        - `𝑞` is the order of the moving average part of the model.
  - `𝑃`, `𝐷` and  `𝑄` denote the same components for the *seasonal* part of the model, with `𝑚` representing the number of periods in each season.

## Python-Library
- Several Python libraries implement ARIMA-SARIMAX models. Four of them are:
    - `statsmodels`: is one of the most complete libraries for statistical modeling in Python. Its API is often more intuitive for those coming from the R environment than for those used to the object-oriented API of scikit-learn.
    - `pmdarima`: This is a wrapper for statsmodels SARIMAX. Its distinguishing feature is its seamless integration with the scikit-learn API, allowing users familiar with `scikit-learn`'s conventions to seamlessly dive into time series modeling.
    - `skforecast`: Among its many forecasting features, it has a new wrapper of statsmodels SARIMAX that also follows the scikit-learn API. This implementation is very similar to that of `pmdarima`, but has been simplified to include only the essential elements for skforecast, resulting in significant speed improvements.
    - `statsForecast`: It offers a collection of widely used univariate time series forecasting models, including automatic ARIMA, ETS, CES, and Theta modeling optimized for high performance using `numba`.


