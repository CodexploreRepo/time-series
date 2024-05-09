# Daily Knowledge

## Day 2

### How to define the baseline model for a time-series

- We can experiment four different baselines:
  - The mean of the entire training set (`hist_mean`)
  - The mean of the last year in the training set (`last_year_mean`)
  - The last known value of the train set (`last`)
  - The naive seasonal forecast (`naive_seasonal`)
- Each baseline was then evaluated on a test set using the MAPE metric
- For the predicting EPS for Johnson & Johnson model,
  - Train-test split:
    - The train set will consist of the data from 1960 to the end of 1979
    - The test set will consist of the four quarters of 1980.
    - The model is to preidct the EPS in four quarters of 1980
  - MAPE on the test set if we use the mean of the entire training set (`hist_mean`) is 70.00%. This means that our baseline deviates by 70% on average from the actual values.
  - MAPE on the test set if we use the mean of the last year in the training set (`last_year_mean`) is reduced to 15.6%. This means that our forecasts deviate from the observed values by 15.6% on average.
    - We can learn from this baseline that future values likely depend on past values that are _not too far back in history_.
      - This is a sign of **autocorrelation**.
  - MAPE on the test set if we use the last known value of the train set (`last`) is 30.45%.
    - This can be explained by the fact that the EPS displays a **cyclical behavior**, where it is high during the first three quarters and then falls at the last quarter.
    - Using the last known value **does not take the seasonality into account**, so we need to use another naive forecasting technique to see if we can produce a better baseline.
  - MAPE on the test set if we use the naive seasonal forecast (`naive_seasonal`) is 11.56%, which is the lowest MAPE from all the baselines.
    - This means that **seasonality has a significant impact on future values**, since repeating the last season into the future yields fairly accurate forecasts
    - Seasonal effects will have to be considered when we develop a more complex forecasting model for this problem.

<p align="center"><img width=400 src="./assets/img/mape_for_baseline_models.png"><br>The MAPE of the four baselines</p>

### Tips

- Modeling (Random Walk): There will be special situations in which a time series can only be forecast using _naive methods_.
  - These are special cases where the process moves at _random_ and _cannot be predicted using statistical learning methods_. This means that we are in the presence of a `random walk`.
- Forecasting feature selection problem: at time making predictions, we cannot use the features that are not available.
  - For example, if we cannot use the temperature as a feature to predict power consumption over the next day as we dont have the value of temperature in the next day
- List of Time-series models: https://unit8co.github.io/darts/
- When creating time-series features (especially rolling or lag), remember to DROP `NaN`

### Backtesting

- Backtesting methods with the use of Scikit-learn's `TimeSeriesSplit`

## Day 1

### Differentiation

- ARIMA assumes the stationarity of the data. If non-stationarity is found, the series should be differenced until stationarity is achieved. This analysis helps to determine the optimal value of the parameter $𝑑$.
- `df.diff()` calculates the difference of a DataFrame element compared with another element in the DataFrame (default is element in previous row).
  - Note: usually go along with dropna `df.diff().dropna()`

### Feature Engineering

#### Time Series Features

- Time series features to see how the trends are impacted by day of week, hour, time of year + holiday factor.
  - Note: for Weekday factor we can assign different weight for different day in a week

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

- Also can include `is_holiday` col

```Python
import holidays
us_holidays = holidays.US()
df['ds'] = df.index
df['isholiday'] = df['ds'].apply(lambda x : x in us_holidays).astype(np.int32)
df.drop(columns = ['ds'], inplace=True)
```

#### Technical Indicator Features

- **Lag** features are added to convert time series forecasting as a supervised Machine Learning problem
  - Lag with respect to a time step $t$ is defined as the values of the series at previous $t$ time steps.
  - For example, lag 1 is the value at time step $t-1$ and lag $m$ is the value at time step $t-m$
- **Rolling (mean, std, min, max)**
- **Rate of Change**

### FB Prophet Model

- Prophet model expects the dataset to be named a specific way.
  - Including the holidays

```Python
from fbprophet import Prophet

# Setup and train model and fit by convert
# .reset_index() to convert datetime index to 'Datetime' col
# 'Datetime' col to 'ds'
# 'Target' col to 'y'
model = Prophet()
model.fit(df.reset_index() \
              .rename(columns={'Datetime':'ds',
                               'Target':'y'}))
"""
including holidays
"""

# Create a dataframe with holiday, ds columns
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

cal = calendar()

df['date'] = df.index.date
df['is_holiday'] = df.date.isin([d.date() for d in cal.holidays()])
holiday_df = df.loc[df['is_holiday']] \
    .reset_index() \
    .rename(columns={'Datetime':'ds'})
# create 'holiday' col in holiday_df and provide what is the holiday name
holiday_df['holiday'] = 'USFederalHoliday'
holiday_df = holiday_df.drop(['is_holiday'], axis=1)
# final holiday_df will have 2 cols only: 'ds' and 'holiday'
holiday_df['ds'] = pd.to_datetime(holiday_df['ds'])

# Init and train model with holidays
model_with_holidays = Prophet(holidays=holiday_df)
```

### Tips

- When dealing with time-series data, it is better to use dates as the index of the dataframe
