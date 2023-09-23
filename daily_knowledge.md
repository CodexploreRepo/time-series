# Daily Knowledge

## Day 1
### Time Series Features
- Time series features to see how the trends are impacted by day of week, hour, time of year

```Python
df['date'] = df.index
df['hour'] = df['date'].dt.hour
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['dayofyear'] = df['date'].dt.dayofyear
df['dayofmonth'] = df['date'].dt.day
df['weekofyear'] = df['date'].dt.weekofyear
```
- **Lag** features are added to convert time series forecasting as a supervised Machine Learning Problem.
### FB Prophet Model
- Prophet model expects the dataset to be named a specific way.
```Python
from fbprophet import Prophet

# Setup and train model and fit by convert 
# .reset_index() to convert datetime index to 'Datetime' col
# 'Datetime' col to 'ds'
# 'Target' col to 'y'
model = Prophet()
model.fit(pjme_train.reset_index() \
              .rename(columns={'Datetime':'ds',
                               'Target':'y'}))

```
### Tips

- When dealing with time-series data, it is better to use dates as the index of the dataframe
