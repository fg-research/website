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
    For simplicity, we will consider only one-month-ahead forecasts. We will use the random
    forest model for predicting next month's inflation based on the current month's values
    of all FRED-MD indicators, including the current month's inflation. We will train the
    model on the FRED-MD time series up to January 2023, and generate the one-month-ahead
    forecasts from February 2023 to January 2024.
    </p>


******************************************
Data
******************************************
.. raw:: html

    <p>
    As discussed in <a href=https://fg-research.com/blog/general/posts/fred-md-overview.html
    target="_blank">our previous post</a>, the FRED-MD dataset is updated on a monthly basis.
    The monthly releases are referred to as <i>vintages</i>. Each vintage includes the data
    from January 1959 up to the previous month. For instance, the 02-2024 vintage contains
    the data from January 1959 to January 2024.
    </p>

    <p>
    The vintages are subject to retrospective adjustments, such as seasonal adjustments,
    inflation adjustments and backfilling of missing values. For this reason, different
    vintages can potentially report different values for the same time series on the
    same date. Furthermore, different vintages can include different time series, as
    indicators are occasionally added and removed from the dataset.
    </p>

    <p>
    We use 02-2023 vintage for training and hyperparameter tuning, while we use the last
    month in each vintage from 03-2023 to 02-2024 for testing. Our approach is different
    from the one used in <a href="#references">[2]</a>, where the same vintage (01-2016)
    is used for both training and testing. In our view, our approach allows us to evaluate
    the model in a more realistic scenario where on a given month we forecast next month's
    inflation using as input the data available on that month, without taking into account
    any ex-post adjustment that could be applied to the data in the future.
    </p>

    <img
        id="inflation-forecasting-random-forest-time-series"
        class="blog-post-image"
        style="width:80%"
        alt="Month-over-month logarithmic change in the US CPI index"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inflation-forecasting-random-forest/time_series_light.png
    />

    <p class="blog-post-image-caption">Month-over-month logarithmic change in the US CPI index (FRED: CPIAUCSL).
    Source: FRED-MD dataset, 02-2024 vintage.</p>

******************************************
Code
******************************************
This section presents and explains the Python code used for the analysis.

==========================================
Set-Up
==========================================
We start by importing the dependencies.

.. code:: python

    import optuna
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import root_mean_squared_error

.. raw:: html

    <br>
    After that, we define a number of auxiliary functions for downloading and processing the FRED-MD dataset.
    As discussed in <a href=https://fg-research.com/blog/general/posts/fred-md-overview.html target="_blank">
    our previous post</a>, the FRED-MD dataset includes a set of transformations to be applied to the time
    series in order to ensure their stationarity, which are implemented in the function below.
    <br>

.. code:: python

    def transform_series(x, tcode):
        '''
        Transform the time series.

        Parameters:
        ______________________________________________________________
        x: pandas.Series
            Time series.

        tcode: int.
            Transformation code.
        '''

        if tcode == 1:
            return x
        elif tcode == 2:
            return x.diff()
        elif tcode == 3:
            return x.diff().diff()
        elif tcode == 4:
            return np.log(x)
        elif tcode == 5:
            return np.log(x).diff()
        elif tcode == 6:
            return np.log(x).diff().diff()
        elif tcode == 7:
            return x.pct_change()
        else:
            raise ValueError(f"unknown `tcode` {tcode}")

.. raw:: html

    <br>
    The function below is used for downloading and processing the training data.
    In this function we download the FRED-MD dataset for the considered vintage,
    transform the time series using the provided transformation codes (with the
    exception of the target time series, for which we use the first order
    logarithmic difference as in <a href="#references">[2]</a>) and define the
    features as the first lag (i.e. the one-month lag) of the all the time series
    (including the target time series). As in <a href="#references">[2]</a>,
    we use the data after January 2060, and we use only the time series without
    missing values.
    <br>


.. code:: python

    def get_training_data(year, month, target_name, target_tcode):
        '''
        Download and process the training data.

        Parameters:
        ______________________________________________________________
        year: int
            The year of the dataset vintage.

        month: int.
            The month of the dataset vintage.

        target_name: string.
            The name of the target time series.

        target_tcode: int.
            The transformation code of the target time series.
        '''

        # get the dataset URL
        file = f"https://files.stlouisfed.org/files/htdocs/fred-md/monthly/{year}-{format(month, '02d')}.csv"

        # get the time series
        data = pd.read_csv(file, skiprows=[1], index_col=0)
        data.columns = [c.upper() for c in data.columns]

        # process the dates
        data = data.loc[pd.notna(data.index), :]
        data.index = pd.date_range(start="1959-01-01", freq="MS", periods=len(data))

        # get the transformation codes
        tcodes = pd.read_csv(file, nrows=1, index_col=0)
        tcodes.columns = [c.upper() for c in tcodes.columns]

        # override the target's transformation code
        tcodes[target_name] = target_tcode

        # transform the time series
        data = data.apply(lambda x: transform_series(x, tcodes[x.name].item()))

        # lag the features
        data = data[[target_name]].join(pd.DataFrame(
            data=data.shift(periods=1).values,
            columns=[c + "_L1" for c in data.columns],
            index=data.index
        ))

        # select the data after January 1960
        data = data[data.index >= pd.Timestamp("1960-01-01")]

        # select the features without missing values
        data = data.loc[:, [target_name] + [c for c in data.columns if c != target_name and data[c].isna().sum() == 0]]

        return data


==========================================
Hyperparameter Tuning
==========================================

==========================================
Model evaluation
==========================================
.. raw:: html

    <img
        id="inflation-forecasting-random-forest-forecasts"
        class="blog-post-image"
        style="width:80%"
        alt="Month-over-month logarithmic change in the US CPI index with random forest (RF) and AR(1) forecasts"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inflation-forecasting-random-forest/forecasts_light.png
    />

    <p class="blog-post-image-caption">Month-over-month logarithmic change in the US CPI index (FRED: CPIAUCSL)
    with random forest (RF) and AR(1) forecasts.</p>

******************************************
References
******************************************

[1] McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for macroeconomic research. *Journal of Business & Economic Statistics*, 34(4), 574-589. `doi: 10.1080/07350015.2015.1086655 <https://doi.org/10.1080/07350015.2015.1086655>`__.

[2] Medeiros, M. C., Vasconcelos, G. F., Veiga, Á., & Zilberman, E. (2021). Forecasting inflation in a data-rich environment: the benefits of machine learning methods. *Journal of Business & Economic Statistics*, 39(1), 98-119. `doi: 10.1080/07350015.2019.1637745 <https://doi.org/10.1080/07350015.2019.1637745>`__.
