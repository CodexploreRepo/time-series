# Date and Time Features

## Date Features

- Date features to see how the trends are impacted by day of week, hour, time of year + holiday factor.
  - Note: for Weekday factor we can assign different weight for different day in a week

```Python
# Note: need to convert the datetime column in the datetime datatype first
df.loc[:, 'datetime'] = pd.to_datetime(df['datetime'])

df['day_of_week'] = df['datetime'].dt.dayofweek
df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month
df['quarter'] = df['datetime'].dt.quarter
df['year'] = df['datetime'].dt.year
# seasonal features: capture recurring patterns in the data
df['day_of_year'] = df['datetime'].dt.dayofyear
df['week_of_year'] = df['datetime'].dt.weekofyear
# sine-cosine features: capturing cyclical patterns such as daily, weekly, or yearly cycles.
df['sin_dayofweek'] = np.sin(2 * np.pi * df['datetime'].dt.dayofweek / 7)
df['cos_dayofweek'] = np.cos(2 * np.pi * df['datetime'].dt.dayofweek / 7)
# temporal features
# month start & end: financial transactions might spike at the end of the month when bills are due,
# or sales might increase at the beginning of the month when people receive their salaries.
df['ismonthstart'] = df['datetime'].dt.is_month_start.astype(int)
df['ismonthend'] = df['datetime'].dt.is_month_end.astype(int)
# quarter start & end: quarterly cycles are important for many business operations, such as financial reporting, sales targets, and inventory management.
# recognizing these periods can help capture trends related to quarterly performance.
df['isquarterstart'] = df['datetime'].dt.is_quarter_start.astype(int)
df['isquarterend'] = df['datetime'].dt.is_quarter_end.astype(int)
# year-end can be important for sales due to holiday shopping
# year-start might see trends related to new year resolutions.
df['isyearstart'] = df['datetime'].dt.is_year_start.astype(int)
df['isyearend'] = df['datetime'].dt.is_year_end.astype(int)
df['isleapyear'] = df['datetime'].dt.is_leap_year.astype(int)
```

- Also can include `is_holiday` column

```Python
import holidays
us_holidays = holidays.US()

df['isholiday'] = df['datetime'].apply(lambda x : x in us_holidays).astype(int)
```

## Time Features

```Python
df['hour'] =  df['datetime'].dt.hour
```

- Express the timestamp in the day as a number of seconds
  - This simply expresses each date in seconds, the number of seconds simply increases linearly with time.

```Python
timestamp_s = pd.to_datetime(df['datetime']).map(datetime.datetime.timestamp) # convert to seconds
```

- **Solution**: to apply sine & cosine transformations to recover the cyclical behavior of time.
- Why we need both sine & consine transformations ?
  - With a single sine transformation, 12 p.m. is equivalent to 12 a.m., and 5 p.m. is equivalent to 5 a.m. This is undesired, as we want to distinguish between morning and afternoon.
  - Cosine is out of phase with the sine function. This allows us to distinguish between 5 a.m. and 5 p.m.

```Python
day = 24 * 60 * 60  # normalise the number of seconds by dividing the total number of seconds per day
# sine transformation
df['day_sin'] = (np.sin(timestamp_s * (2*np.pi/day))).values
# cosine transformation
df['day_cos'] = (np.cos(timestamp_s * (2*np.pi/day))).values
```
