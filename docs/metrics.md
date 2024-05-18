# Metrics

## Mean Absolute Percentage Error (MAPE)

- **Mean Absolute Percentage Error (MAPE)**: a measure of prediction accuracy for forecasting methods that is easy to interpret and **independent of the scale of our data** (either two-digit values or six-digit values), the MAPE will always be expressed as a percentage.
- MAPE _returns the percentage of how much the forecast values deviate from the observed_ or actual values on average, whether the prediction was higher or lower than the observed values
  - For example: MAPE of 30.00%. This means that our baseline deviates by 30% on average from the actual values.
- The MAPE is defined in equation, where:
  - $A_i$ is the actual value at point $i$ in time
  - $F_i$ is the forecast value at point $i$ in time
  - $n$ is simply the number of forecasts.
    $$ \text{MAPE} = \frac {1}{n} \sum_i{\left\lvert{\frac{A_i-F_i}{A_i}}\right\rvert}$$
- Drawback: we cannot use MAPE if the series contains 0-value it is impossible to calculate the percentage difference from an observed value of 0 because that implies a division by 0.

## Mean Squared Error (MSE)

- When to use MSE ?
  - In case the series contains 0-value, so we are unable to use MAPE, so MSE is a good option.
  - In case the prediction range is small (0.1 or 1), and we want to amplify the error
- How to know if the MSE is good or bad ?
  - For example, for the random walk series with the range varies from -30 to 30. The best forecast produces the MSE exceeds 300. This is an extremely high value considering that our random walk dataset does not exceed the value of 30.

## Mean Absolute Error (MAE)

- `mean_absolute_error` metrics is easy to interpret, as it returns the average of the absolute difference between the predicted and actual values, instead of a squared difference like the **MSE**.
- For example, `MAE=2765` means that the actual prediction will be either above or below the actual value around $2765
