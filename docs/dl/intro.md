# Introduction for Large-scale Forecasting with Deep Learning

## Why & When using Deep Learning for forecasting ?

- Why: Statistical models have their limitations, especially when a dataset is **large** and has many features and **non-linear** relationships.
- When:
  - A dataset is considered to be large when we have **more than 10,000 data points**.
  - If your data has **multiple seasonal** periods, the SARIMAX model cannot be used.
    - For example, suppose you must forecast the hourly temperature.
      - There is a daily seasonality, as temperature tends to be lower at night and higher during the day
      - There is also a yearly seasonality, due to temperatures being lower in winter and higher during summer.

## Types of Deep Learning Models

- Main types of deep learning models:
  - **Single-step** model: represents the forecast of one variable one step into the future
  - **Multi-step** model: forecasts many timesteps into the future
    - For example: given hourly data, we may want to forecast the next 24 hours.
  - **Multi-output** model: generates predictions for more than one target.
    - For example: the model to forecast the temperature and wind speed
