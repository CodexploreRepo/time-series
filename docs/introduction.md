# Introduction
## Time Series Patterns
- A typical time series model can exhibits different patterns. 
<p align="center"><img src="../assets/img/time-series-pattern.webp"></p>

- A time series can be a graph of a combination of all of above patterns . Therefor it is important to understand *components of a time series* in detail 
## Components of a time series
- A time series it is composed of **Trend (T)** , **Seasonality (S)**, **Cyclic (C)** and **Residual (R)** components.

$$ U_t = T_t + S_t + C_t + R_t $$

1. *Trend Component*: The long-term tendency of a series to increase or fall (upward trend or downward trend).

2. *Seasonality Component*: The periodic fluctuation in the time series within a certain period. These fluctuations form a pattern that tends to repeat from one seasonal period to the next one.

3. *Cycles Component*: Long departures from the trend due to factors others than seasonality. Cycles usually occur along a large time interval, and the lengths of time between successive peaks or troughs of a cycle are not necessarily the same.

4. *Irregular movement Component*: The movement left after explaining the trend, seasonal and cyclical movements; random noise or error in a time series.