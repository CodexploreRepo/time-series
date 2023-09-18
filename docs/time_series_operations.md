# Time Series Operation

## Resampling

- **Resampling** means representing the data with different frequency
  - **down-sampling** convert to lower frequency
  - can be done using
    - `resample()` aggregates data based on specified freq and aggre function
    - `asfreq()` select data based on specified freq and returns the value at the end of specified interval
- Assume a temp sensor takes measurement every minute, and if we do not need to have minute-level precision, we can take the average of 60 mins measurement

## Shifting

- Time series data analysis require shift data points to make a comparison
  - `shift` shifts the data
  - `tshift` shifts the time index

## Rolling

- Rolling is a very useful operation for time series data.
- Rolling means creating a rolling window with a specified size and perform calculations on the data in this window which rolls through the data
- For example: `rolling_window = 5`, the first 4 records will be `NaN` and only starting 5th record has value which is the mean of the first 5 records.
