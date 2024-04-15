# Time Series Operation

## Resampling

- **Resampling** means representing the data with different frequency
  - **down-sampling** convert to lower frequency
  - can be done using
    - `resample()` aggregates data based on specified freq and aggre function
    - `asfreq()` select data based on specified freq and returns the value at the end of specified interval
      - `df.asfreq('MS')` 'MS' stands for "Month Start" frequency. This method allows you to convert the frequency of your time series data to monthly data starting at the beginning of each month.
- Assume a temp sensor takes measurement every minute, and if we do not need to have minute-level precision, we can take the average of 60 mins measurement

## Shifting (Lag)

- Lag with respect to a time step $t$ is defined as the values of the series at previous $t$ time steps.
  - For example, lag 1 is the value at time step $t-1$ and lag $m$ is the value at time step $t-m$
- Time series data analysis require shift data points to make a comparison
  - `shift` shifts the data
    - For example: `df['price'].shift(2)` means that the first two records will be `NaN` (or empty), only starting from the 3rd record will have the value of the first one (i.e. lag=2)

## Rolling (mean, std, min, max)

- Rolling is a very useful operation for time series data.
- Rolling means creating a rolling window with a specified size and perform calculations on the data in this window which rolls through the data
- For example: `rolling_window = 5`, the first 4 records will be `NaN` and only starting 5th record has value which is the mean of the first 5 records.

```Python
# As the data is hourly, so window size can be 6, 12, 24
for window in [6,12,24]:
  df[f'pjme_{window}_hrs_lag'] = df['PJME_MW'].shift(window)

  df[f'pjme_{window}_hrs_rolling_mean'] = df['PJME_MW'].rolling(window = window).mean()
  df[f'pjme_{window}_hrs_rolling_std'] = df['PJME_MW'].rolling(window = window).std()
  df[f'pjme_{window}_hrs_rolling_max'] = df['PJME_MW'].rolling(window = window).max()
  df[f'pjme_{window}_hrs_rolling_min'] = df['PJME_MW'].rolling(window = window).min()
```

## Percentage Change (Rate of Change)

- Rate of Change is to calculate how much percentage a variable has changed over a time period
  - Formula: rate*of_change (roc) $= (y_t - y*{t-1}) /y\_{t-1}$
- `df['col'].pct_change()` default `period = 1`
