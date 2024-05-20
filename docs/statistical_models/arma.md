# Autoregressive Moving Average (ARMA)

- The autoregressive moving average model, denoted as $ARMA(p,q)$, is the combination of the autoregressive model $AR(p)$ and the moving average model $MA(q)$.
- An ARMA(p,q) process will display a **decaying pattern** or a **sinusoidal** pattern on both the ACF and PACF plots.
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
