# Time Series

## Basics

- [Time Series](./docs/introduction.md)
- [Time Series Operations](./docs/time_series_operations.md)
- [Time Series Forecasting](./docs/time_series_forecasting.md)

## EDA

- [Stationary](./docs/eda/eda_stationary.md)
- [Autocorrelation](./docs/eda/eda_autocorrelation.md)
- [Time Series Decomposition](./docs/eda/eda_time-series-decomposition.md)

## Feature Engineering

- [Basics](./docs/fe/basics.md):
  - Lag
  - Rolling, Expanding: `mean`, `std`, `min`, `max`
  - Rate of Change
- [Date & Time Features](./docs/fe/date_and_time_features.md)
- [Technical Indicators](./docs/fe/technical_indicators.md)
  - **Simple Moving Average (SMA)** - (a.k.a _rolling mean_): overall trend direction
  - **Exponential Moving Average (EMA)**: short-term trend as EMA gives more weightage to recent data points.
  - **Moving Average Convergence Divergence (MACD)**: to identify shifts in market momentum and potential breakout points
  - **Relative Strength Index (RSI)**: measures the speed and change of price movements
  - **Bollinger Bands**: measure the volatility of a market and identify potential overbought or oversold conditions.

## Modelling

- [Baseline Model](./docs/baseline_model.md)
- [Random Walk](./docs/random_walk.md)
- [Metrics](./docs/metrics.md)

### Statistical Forecasting Models

- [Introduction](./docs/statistical_models/intro.md)

#### Forecasting stationary series

- The below models only can be used for **stationary** time series, or those non-stationary series which require only **one round** of transformations, mainly differencing to make it stationary. The forecasts from each model _returned differenced values_, which required us to reverse this transformation in order to bring the values back to the scale of the original data.
  - [Moving Average MA(q) model](./docs/statistical_models/moving_average.md)
  - [Autoregressive AR(p) model](./docs/statistical_models/autoregressive.md)
  - [Autoregressive Moving Average ARMA(p,q) model](./docs/statistical_models/arma.md)

#### Forecasting non-stationary time series

- We can forecast **non-stationary** time series & avoid the steps of modeling on stationary (differenced) data and having to inverse transform the forecasts by adding the **integration order** component ($d >= 2$), which is denoted by the variable $d$ into the $ARMA(p,q)$ model
  - [Autoregressive Integrated Moving Average ARIMA(p,d,q) model](./docs/statistical_models/arima.md)
  - [Seasonal Autoregressive Integrated Moving Average SARIMA(p,d,q)(P,D,Q)m model](./docs/statistical_models/sarima.md)
  - [SARIMA with Exogenous Variables (X)]
  - [ARIMA & S-ARIMA-X](./docs/statistical_models/arima_sarimax.md)

### ML Models

- [Forecasting with Gradient Boosting Models]

## How-to Guide

- Create conda env: `conda create -n time-series`
- Activate conda env: `conda activate time-series`
- Install dependencies: `conda install --file requirements.txt`

## Others

- [Glossary](./docs/glossary.md)
