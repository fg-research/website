.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Forecasting US inflation with random forests
   :keywords: Time Series, Forecasting, Machine Learning, Macroeconomics, Inflation

######################################################################################
Forecasting US inflation with random forests
######################################################################################

.. raw:: html

    <p>
    Inflation forecasts are used for informing economic decisions at various levels,
    from households to businesses and policymakers. Machine learning approaches
    offer several advantages for inflation forecasting, including the ability to
    handle large and complex datasets, capture nonlinear relationships, and adapt
    to changing economic conditions.
    </p>

    <p>
    Several recent papers have studied the problem of forecasting US inflation with
    machine learning methods using the <a href="https://research.stlouisfed.org/econ/mccracken/fred-databases/" target="_blank">
    FRED-MD</a> dataset <a href="#references">[1]</a>. The FRED-MD dataset includes
    over 100 monthly time series belonging to 8 different groups of US macroeconomic
    indicators: output and income, labour market, consumption and orders, orders and
    inventory, money and credit, interest rates and exchange rates, prices, and stock
    market. For a detailed overview of the FRED-MD dataset, we refer to
    <a href=https://fg-research.com/blog/general/posts/fred-md-overview.html
    target="_blank">our previous post</a>.
    </p>

    <p>
    In this post, we will focus on the random forest model introduced in <a href="#references">[2]</a>,
    which was found to outperform both standard univariate forecasting models such as the AR(1) model
    and several other machine learning methods including Lasso, Ridge and Elastic Net regressions.
    We will use the random forest model for forecasting the US CPI monthly inflation, which we
    define as the month-over-month logarithmic change in the US CPI index as in <a href="#references">[2]</a>.
    </p>

    <p>
    For simplicity, we will consider only one-month-ahead forecasts. We will train the model
    to predict next month's inflation based on the current month's values of all FRED-MD
    time series, including the current month's inflation. We will train the model
    on the FRED-MD time series up to January 2023, and generate the one-month-ahead forecasts
    from February 2023 to January 2024.
    </p>

******************************************
Model
******************************************

******************************************
Data
******************************************
.. raw:: html

    <p>
    As discussed in <a href=https://fg-research.com/blog/general/posts/fred-md-overview.html
    target="_blank">our previous post</a>, the FRED-MD dataset is updated on a monthly basis.
    The monthly releases are referred to as <i>real-time vintages</i>. Each real-time vintage
    includes the data from January 1959 up to the previous month. For instance, the 02-2024
    vintage contains the data from January 1959 to January 2024.
    </p>

    <p>
    The real-time vintages are subject to ex-post adjustments, such as seasonal adjustments,
    inflation adjustments and backfilling of missing values. For this reason, different
    real-time vintages can potentially report different values for the same time series
    on the same date. Furthermore, different real-time vintages can include different
    time series, as indicators are occasionally added or removed from the dataset.
    </p>

    <p>
    The real-time vintages also include the suggested transformations to be applied to the time
    series in order to ensure their stationarity. The possible transformations are first order
    difference, second order difference, logarithm, first order logarithmic difference,
    second order logarithmic difference, percentage change, and no transformation.
    </p>

******************************************
Code
******************************************

==========================================
Set-Up
==========================================

==========================================
Hyperparameter Tuning
==========================================

==========================================
Model evaluation
==========================================

******************************************
References
******************************************

[1] McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for macroeconomic research. *Journal of Business & Economic Statistics*, 34(4), 574-589. `doi: 10.1080/07350015.2015.1086655 <https://doi.org/10.1080/07350015.2015.1086655>`__.

[2] Medeiros, M. C., Vasconcelos, G. F., Veiga, Á., & Zilberman, E. (2021). Forecasting inflation in a data-rich environment: the benefits of machine learning methods. *Journal of Business & Economic Statistics*, 39(1), 98-119. `doi: 10.1080/07350015.2019.1637745 <https://doi.org/10.1080/07350015.2019.1637745>`__.
