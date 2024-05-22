# Time Series

- [Time Series](./docs/introduction.md)
- [Time Series Operations](./docs/time_series_operations.md)
- [Time Series Forecasting](./docs/time_series_forecasting.md)

## Modelling

- [Baseline Model](./docs/baseline_model.md)
- [Random Walk](./docs/random_walk.md)
  - [EDA: Stationary](./docs/eda/eda_stationary.md)
  - [EDA: Autocorrelation](./docs/eda/eda_autocorrelation.md)
- [Metrics](./docs/metrics.md)

### Statistical Forecasting Models

- [Introduction](./docs/statistical_models/intro.md)

#### Forecasting stationary series

- The below models only can be used for **stationary** time series, or those non-stationary series which require only **one round** of transformations, mainly differencing to make it stationary. The forecasts from each model _returned differenced values_, which required us to reverse this transformation in order to bring the values back to the scale of the original data.
  - [Moving Average $MA(q)$ model](./docs/statistical_models/moving_average.md)
  - [Autoregressive $AR(p)$ model](./docs/statistical_models/autoregressive.md)
  - [Autoregressive Moving Average $ARMA(p,q)$ model](./docs/statistical_models/arma.md)

#### Forecasting non-stationary time series

- We can forecast **non-stationary** time series & avoid the steps of modeling on stationary (differenced) data and having to inverse transform the forecasts by adding the **integration order** component ($d >= 2$), which is denoted by the variable $d$ into the $ARMA(p,q)$ model
  - [Autoregressive Integrated Moving Average $ARIMA(p,d,q)$ model](./docs/statistical_models/arima.md)
  - [ARIMA & S-ARIMA-X](./docs/statistical_models/arima_sarimax.md)

### ML Models

- [Forecasting with Gradient Boosting Models]

## How-to Guide

- Create conda env: `conda create -n time-series`
- Activate conda env: `conda activate time-series`
- Install dependencies: `conda install --file requirements.txt`

## Others

- [Glossary](./docs/glossary.md)
