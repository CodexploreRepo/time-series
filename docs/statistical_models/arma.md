# Autoregressive Moving Average (ARMA)

- The autoregressive moving average model, denoted as $ARMA(p,q)$, is the combination of the autoregressive model $AR(p)$ and the moving average model $MA(q)$.
- An ARMA(p,q) process will display a **decaying pattern** or a **sinusoidal** pattern on _both the ACF and PACF_ plots.
  - Therefore, they cannot be used to estimate the orders $p$ and $q$.
- The **general modeling procedure** does not rely on the ACF and PACF plots. Instead, we fit many ARMA(p,q) models and perform model selection and residual analysis.
  - **Model selection** is done with the Akaike information criterion (AIC).
    - AIC quantifies the information loss of a model, and it is related to the number of parameters in a model and its goodness of fit. _The lower the AIC, the better the model_.
  - Residual analysis on the best model selected based on model selection
    - The **Q-Q plot** is a graphical tool for comparing two distributions. We use it to compare the distribution of the _residuals_ against a theoretical _normal_ distribution.
      - If the plot shows a straight line that lies on y = x, then both distributions are similar.
      - Otherwise, it means that the residuals are not normally distributed.
    - The **Ljung-Box** test allows us to determine whether the residuals are _correlated_ or not.
      - The null hypothesis states that the data is independently distributed and uncorrelated.
        - If the returned p-values are larger than 0.05, we cannot reject the null hypothesis, meaning that the residuals are uncorrelated, just like white noise.

## Introduction

- The autoregressive moving average process $ARMA(p,q)$ is a combination of the autoregressive process and the moving average process.
- The $ARMA(p,q)$ process is expressed as a linear combination of
  - On its own previous values $y_{t-p}$ and a constant $C$, just like in an _autoregressive_ process
  - On the mean of the series $\mu$, the current error term $\epsilon_t$, and past error terms $\epsilon_{t-q}$, like in a _moving average_ process.

$$y_t = C + \epsilon_t + \varphi_1y_{t–1} + \varphi_2y_{t–2} +⋅⋅⋅+ \varphi_p y_{t–p} + \mu + \epsilon_t + \theta_1\epsilon_{t-1} + ... + \theta_q\epsilon_{t-q}$$

### Simulating an ARMA process

- Simulate an ARMA(1,1) process with the below equation. - This is equivalent to combining an MA(1) process with an AR(1) process.
  $$y_t = 0.33y_{t-1} + 0.9\epsilon_{t-1} + \epsilon_t$$
- Use the `ArmaProcess` function from the `statsmodels` library to simulate our ARMA(1,1) process.
  - $AR(1)$ process will have a coefficient of 0.33.
    - However, the function expects to have the coefficient of the autoregressive process with its opposite sign, so it is –0.33.
  - $MA(1)$ process will have a coefficient is 0.9
  - **Note**: when defining your arrays of coefficients, the first coefficient is always equal to 1, as specified by the library, which represents the coefficient at lag 0.

```Python
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess

ar1 = np.array([1, -0.33])
ma1 = np.array([1, 0.9])

ARMA_1_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=1000)
```

## Identifying a stationary ARMA process

<p align="center"><img src="../../assets/img/arma-identification-framework.png" height=700><br>Steps to identify a random walk, a moving average process MA(q), an autoregressive process AR(p), and an autoregressive moving average process ARMA(p, q)</p>

- If neither of the ACF and PACF plots shows a clear cutoff between significant and non-significant coefficients, then we have an ARMA(p,q) process.
