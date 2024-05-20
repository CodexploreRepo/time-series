# Autoregressive Integrated Moving Average (ARIMA)

- The $ARIMA(p,d,q)$ model can be applied on non-stationary time series and has the added advantage of returning forecasts in the same scale as the original series.
- The order of integration $d$ defines how many times a series must be differenced to become stationary.
  - This parameter then allows us to fit the model on the original series and get a forecast in the same scale, unlike the ARMA(p,q) model, which required the series to be stationary using transformations (differencing) for the model to be applied and required to reverse the transformations on the forecasts.
- To apply the ARIMA(p,d,q) model, we added an extra step to our **general modeling procedure**, which simply involves finding the value for the order of integration $d$.

## Introduction

- An autoregressive integrated moving average $ARIMA(p,d,q)$ process is the combination of
  - Autoregressive process $AR(p)$
  - Integration $I(d)$
  - Moving average process $MA(q)$.
- ARIMA model is simply an ARMA model that can be applied on **non-stationary** time series.
  - The ARMA(p,q) model requires the series to be stationary **before** fitting an ARMA(p,q) model
  - The ARIMA(p,d,q) model can be used on **non-stationary** series.
- ARIMA process uses the differenced series ($y_t'$) instead of using the original series ($y_t$)
  - **Note**: that $y_t'$ can represent a series that has been differenced more than once.
- The general equation of the ARIMA process:

$$y_t' = C + \varphi_1y_{t–1}' + \varphi_2y_{t–2}' +⋅⋅⋅+ \varphi_p y_{t–p}' + \epsilon_t + \theta_1\epsilon_{t-1}' + ... + \theta_q\epsilon_{t-q}'$$
