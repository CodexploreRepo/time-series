# SARIMA with Exogenous Variables (X)

- The SARIMAX model allows you to include external variables, also termed exogenous variables, to forecast your target.
- Transformations (differencing) are applied only on the target variable, not on the exogenous variables.
- Problem with Exogenous variables: SARIMAX is not recommended to forecast multiple timesteps into the future as the exogenous variables might not be available multiple steps in the future, so they must also be forecasted.
  - This can magnify the errors on the final forecast.
  - To avoid that, you must predict only the next timestep using the rolling forecast function.

## Introduction

- The $\text{SARIMAX}$ model further extends the $\text{SARIMA}(p,d,q)(P,D,Q)_m$ model by adding the effect of exogenous variables $X$.
- The SARIMAX model is the **most general model** for forecasting time series.
  - If no seasonal patterns, it becomes an ARIMAX model.
  - With no exogenous variables, it is a SARIMA model.
  - With no seasonality or exogenous variables, it becomes an ARIMA model.
- SARIMA**X** express the present value $y_t$ simply as a SARIMA model to which we add any number $n$ of exogenous variables $X_t$ as below equation:
  - In other words, the SARIMA**X** model simply adds a linear combination of exogenous variables to the SARIMA model.

$$y_t =\text{SARIMA}(p,d,q)(P,D,Q)_m + \sum_{i=1}^{n}{\beta_i X_t^i}$$

## Exploring the exogenous variables of the US macroeconomics dataset

- There are two ways to work with exogenous variables for time series forecasting.
  - Method 1: we could train multiple models with various combinations of exogenous variables, and see which model generates the best forecasts.
  - Method 2: we can simply include all exogenous variables and stick to model selection using the AIC, as we know this yields a good-fitting model that does not overfit.
- All variables in the US macroeconomics dataset

|  Variable  |                               Description                                |
| :--------: | :----------------------------------------------------------------------: |
| `realgdp`  | Real gross domestic product (the target variable or endogenous variable) |
| `realcons` |                  Real personal consumption expenditure                   |
| `realinv`  |                  Real gross private domestic investment                  |
| `realgovt` |           Real federal consumption expenditure and investment            |
| `realdpi`  |                      Real private disposable income                      |
|   `cpi`    |             Consumer price index for the end of the quarter              |
|    `m1`    |                          M1 nominal money stock                          |
| `tbilrate` |      Quarterly monthly average of the monthly 3-month treasury bill      |
|  `unemp`   |                            Unemployment rate                             |
|   `pop`    |                Total population at the end of the quarter                |
|   `infl`   |                              Inflation rate                              |
| `realint`  |                            Real interest rate                            |

```Python
macro_econ_data = sm.datasets.macrodata.load_pandas().data
macro_econ_data.tail()
```

|     | year | quarter | realgdp | realcons | realinv | realgovt | realdpi |     cpi |     m1 | tbilrate | unemp |     pop |  infl | realint |
| --: | ---: | ------: | ------: | -------: | ------: | -------: | ------: | ------: | -----: | -------: | ----: | ------: | ----: | ------: |
| 198 | 2008 |       3 | 13324.6 |   9267.7 | 1990.69 |  991.551 |  9838.3 | 216.889 | 1474.7 |     1.17 |     6 |  305.27 | -3.16 |    4.33 |
| 199 | 2008 |       4 | 13141.9 |   9195.3 | 1857.66 |  1007.27 |  9920.4 | 212.174 | 1576.5 |     0.12 |   6.9 | 305.952 | -8.79 |    8.91 |
| 200 | 2009 |       1 | 12925.4 |   9209.2 | 1558.49 |  996.287 |  9926.4 | 212.671 | 1592.8 |     0.22 |   8.1 | 306.547 |  0.94 |   -0.71 |
| 201 | 2009 |       2 | 12901.5 |     9189 | 1456.68 |  1023.53 | 10077.5 | 214.469 | 1653.6 |     0.18 |   9.2 | 307.226 |  3.37 |   -3.19 |
| 202 | 2009 |       3 | 12990.3 |     9256 |  1486.4 |  1044.09 | 10040.6 | 216.385 | 1673.9 |     0.12 |   9.6 | 308.013 |  3.56 |   -3.44 |

## Problem with Exogenous variables

- What if you wish to predict two timesteps into the future?
  - While this is possible with a SARIMA model, the SARIMAX model requires us to forecast the exogenous variables too.
- The only way to avoid that situation is to predict only one timestep into the future and **wait to observe** the exogenous variable before predicting the target for another timestep into the future.
- Summary: There is no clear recommendation to predict only one timestep.
  - If you determine that your exogenous variable can be accurately predicted, you can recommend forecasting many timesteps into the future.
  - Otherwise, your recommendation must be to predict one timestep at a time and justify your decision by explaining that errors will accumulate as more predictions are made, meaning that the forecasts will lose accuracy.

## Forecasting with SARIMAX

- In this example, we will use SARIMAX to forecast the real GDP with the exploration of exogenous variables: 'realcons', 'realinv', 'realgovt', 'realdpi', 'cpi'

```Python
target = macro_econ_data['realgdp']
exog = macro_econ_data[['realcons', 'realinv', 'realgovt', 'realdpi','cpi']] # exogenous variables
```

- **Step 1**: Check for stationarity and apply transformation in order to set the parameter $d$ & $D$

```Python
def check_stationarity(series, p_significant=0.05):
    # Copied from https://machinelearningmastery.com/time-series-data-stationary-python/

    adfuller_result = adfuller(series)
    kpss_result = kpss(series)

    print(f'ADF Statistic : {adfuller_result[0]:.5f}, p-value: {adfuller_result[1]:.5f}')
    print('Critical Values:')
    for key, value in adfuller_result[4].items():
        print('\t%s: %.3f' % (key, value))
    print(f'KPSS Statistic: {kpss_result[0]:.5f}, p-value: {kpss_result[1]:.5f}')
    if (adfuller_result[1] <= p_significant) & (adfuller_result[4]['5%'] > adfuller_result[0]) & (kpss_result[1] > p_significant):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")

check_stationarity(target) # non-stationary
# The time series is not stationary.
# Apply a transformation to our data in order to make it stationary. Let’s apply a first-order differencing using numpy.
# apply a transformation and test for stationarity again.
target_diff = target.diff()
check_stationarity(target_diff[1:])
```

- Since the series is stationary after the first difference, so $d = 1$ and we do not need to take a seasonal difference to make the series stationary, so $D = 0$.

- **Step 2**: Time Series Decomposition to identify if there is a seasonality in the series

```Python
# Decompose the series using the STL function.

def decompose_ts(df, period=12):
    decomposition = STL(df, period=period).fit()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(15,8))

    ax1.plot(decomposition.observed)
    ax1.set_ylabel('Observed')

    ax2.plot(decomposition.trend)
    ax2.set_ylabel('Trend')

    ax3.plot(decomposition.seasonal)
    ax3.set_ylabel('Seasonal')

    ax4.plot(decomposition.resid)
    ax4.set_ylabel('Residuals')

    plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))

    fig.autofmt_xdate()
    plt.tight_layout()
# The period is equal to the frequency m. Since we have quarterly data, the period is 4.
decompose_ts(target, period=4)
```

- From the time-series decomposition, the series does not contain the seasonality.
- However, we still can try to with SARIMAX model, usually P, Q willl return 0.
- **Step 3**: Model selection with AIC to determine $p$, $q$, $P$ and $Q$
  - Note: The implementation of SARIMA in statsmodels simply uses $s$ instead of $m$ — they both denote the frequency.

```Python
def optimize_SARIMAX(endog: Union[pd.Series, list],
                    exog: Union[pd.Series, list],
                    order_list: list, d: int, D: int, s: int) -> pd.DataFrame:

    results = []

    for order in tqdm(order_list):
        try:
            model = SARIMAX(
                endog,
                exog,
                order=(order[0], d, order[1]),
                seasonal_order=(order[2], D, order[3], s),
                simple_differencing=False).fit(disp=False)
        except:
            continue

        aic = model.aic
        results.append([order, aic])

    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q,P,Q)', 'AIC']

    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)

    return result_df
```

- Since the data is collected quarterly, $m = 4$ (or $s$ in `SARIMAX` model in `statsmodels`)

```Python
# p,q, P,Q and m
p = range(0, 4, 1)
d = 1
q = range(0, 4, 1)
P = range(0, 4, 1)
D = 0
Q = range(0, 4, 1)
s = 4    # m=4 as the data is collect quarterly

parameters = product(p, q, P, Q)
parameters_list = list(parameters)


# Train-test split
# use the first 200 instances of both the target and exogenous variables
target_train = target[:200]
exog_train = exog[:200]

result_df = optimize_SARIMAX(target_train, exog_train, parameters_list, d, D, s)

# 	   (p,q,P,Q)	   AIC
# 0	(3, 3, 0, 0)	1742.821107
# 1	(3, 3, 1, 0)	1744.967747
# 2	(3, 3, 0, 1)	1744.996645
# 3	(2, 2, 0, 0)	1745.488067
# 4	(3, 3, 2, 2)	1746.287115
```

- The function returns the verdict that the $\text{SARIMAX}(3,1,3)(0,0,0)_4$ model is the model with the lowest AIC.
- As expected, the seasonal component of the model has only orders of 0.

```Python
# fit the SARIMAX with the best hyper-parameters and perform the Residual Analysis
best_model = SARIMAX(target_train, exog_train, order=(3,1,3), seasonal_order=(0,0,0,4), simple_differencing=False)
best_model_fit = best_model.fit(disp=False)
print(best_model_fit.summary())

#                                SARIMAX Results
# ==============================================================================
# Dep. Variable:                realgdp   No. Observations:                  200
# Model:               SARIMAX(3, 1, 3)   Log Likelihood                -859.411
# Date:                Sat, 25 May 2024   AIC                           1742.821
# Time:                        16:11:00   BIC                           1782.341
# Sample:                             0   HQIC                          1758.816
#                                 - 200
# Covariance Type:                  opg
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# realcons       0.9688      0.045     21.600      0.000       0.881       1.057
# realinv        1.0136      0.033     30.816      0.000       0.949       1.078
# realgovt       0.7286      0.127      5.741      0.000       0.480       0.977
# realdpi        0.0096      0.025      0.387      0.699      -0.039       0.058
# cpi            5.8679      1.304      4.499      0.000       3.312       8.424
# ar.L1          1.0650      0.398      2.677      0.007       0.285       1.845
# ar.L2          0.4920      0.700      0.703      0.482      -0.880       1.864
# ar.L3         -0.6744      0.336     -2.005      0.045      -1.334      -0.015
# ma.L1         -1.1040      0.430     -2.570      0.010      -1.946      -0.262
# ma.L2         -0.3226      0.767     -0.421      0.674      -1.825       1.180
# ma.L3          0.6480      0.403      1.608      0.108      -0.142       1.438
# sigma2       330.1999     30.572     10.801      0.000     270.281     390.119
```

- All exogenous variables have a p-value smaller than 0.05, except for `realdpi`, which has a p-value of 0.712.
  - This means that the coefficient of `realdpi` is not significantly different from 0, and in fact, the **coefficient** of `realdpi` is 0.0096

```Python
# Residuals Analysis
best_model_fit.plot_diagnostics(figsize=(10,8));

residuals = best_model_fit.resid
jb_df = acorr_ljungbox(residuals, np.arange(1, 11, 1))
(jb_df["lb_pvalue"] >= 0.05).sum() == 10 # True: this to ensure all the lags, the p-value exceed 0.05, so we cannot reject the null hypo
```

- From the above plots, the distribution of residuals is very close to a normal distribution.
- All the p-values are greater than 0.05. Therefore, we do not reject the null hypothesis, and we conclude that the residuals are independent and uncorrelated.
- Our model has passed all the tests from the residuals analysis, and we are ready to use it for forecasting.
- **Step 4**: Forecasting
  - As mentioned before, the caveat of using a SARIMA**X** model is that it is reasonable to predict **only** the next timestep, as if the exogenous variables might not be available if we predict in a long horizon, so the SARIMAX will have to predict the exogenous variables as well if they are not available, which would lead us to accumulate prediction errors in the final forecast.
  - Hence, we will use the rolling forecast function `rolling_forecast`
  - Baseline model: last known value

```Python
def rolling_forecast(endog: Union[pd.Series, list],
                     exog: Union[pd.Series, list],
                     train_len: int, horizon: int, window: int,method: str) -> list:

    total_len = train_len + horizon

    if method == 'last':
        pred_last_value = []

        for i in range(train_len, total_len, window):
            last_value = endog[:i].iloc[-1]
            pred_last_value.extend(last_value for _ in range(window))

        return pred_last_value

    elif method == 'SARIMAX':
        pred_SARIMAX = []

        for i in range(train_len, total_len, window):
            model = SARIMAX(endog[:i], exog[:i], order=(3,1,3), seasonal_order=(0,0,0,4), simple_differencing=False)
            res = model.fit(disp=False)
            predictions = res.get_prediction(exog=exog)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_SARIMAX.extend(oos_pred)

        return pred_SARIMAX

target_train = target[:196]
target_test = target[196:]

TRAIN_LEN = len(target_train)
HORIZON = len(target_test)
WINDOW = 1

pred_last_value = rolling_forecast(target, exog, TRAIN_LEN, HORIZON, WINDOW, 'last')
pred_SARIMAX = rolling_forecast(target, exog, TRAIN_LEN, HORIZON, WINDOW, 'SARIMAX')

pred_df = pd.DataFrame({'actual': target_test})
pred_df['pred_last_value'] = pred_last_value
pred_df['pred_SARIMAX'] = pred_SARIMAX

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_last = mape(pred_df.actual, pred_df.pred_last_value)
mape_SARIMAX = mape(pred_df.actual, pred_df.pred_SARIMAX)
# 0.736849498653785 0.7025002590225526
```

- The SARIMAX model is the winning model by only 0.04%
