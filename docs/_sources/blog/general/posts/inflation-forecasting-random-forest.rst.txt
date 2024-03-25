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
    from households to businesses and policymakers. The application of machine learning
    methods to inflation forecasting offers several potential advantages, including
    the ability to handle large and complex datasets, capture nonlinear relationships,
    and adapt to changing economic conditions.
    </p>

    <p>
    Several recent papers have studied the performance of machine learning models
    for forecasting US inflation using the <a href="https://research.stlouisfed.org/econ/mccracken/fred-databases/" target="_blank">
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
    which was found to outperform both standard univariate forecasting models such as the autoregressive (AR) model
    and several other machine learning methods including Lasso, Ridge and Elastic Net regressions.
    We will use the random forest model for forecasting the US CPI monthly inflation, which we
    define as the month-over-month logarithmic change in the US CPI index as in <a href="#references">[2]</a>.
    </p>

    <p>
    For simplicity, we will consider only one-month-ahead forecasts. We will use the random
    forest model for predicting next month's inflation based on the past values
    of all FRED-MD indicators, including the past inflation. We will evaluate the model using
    an expanding window approach: on each month we will train the model using all the available
    data up to that month, and generate the forecast for next month. We use the data
    from January 2015 to January 2024 for our analysis. Our findings indicate that,
    over the considered time period, the random forest model outperforms the AR model
    under different measures of forecast error.
    </p>

******************************************
Model
******************************************
.. raw:: html

    <p>
    A random forest is an ensemble of decision trees. A decision tree approximates the
    relationship between the target and the features by splitting the feature space
    into different subsets, and generating a constant prediction for each subset.
    In the regression case, which is the one relevant to this post, the constant
    prediction is the average of the target values in the subset.
    A decision tree can be seen as a nonparametric regression model,
    where the regression function that links the target to the features is estimated
    using a piecewise constant approximation.
    </p>

    <p>
    The partition of the feature space is determined in a recursive manner during the
    process of growing the tree. At the beginning of this process, the tree
    contains only one node, referred to as root note, which includes the full
    dataset. In the root node, the target is predicted with the average of all target observations in the dataset.
    After that, the dataset is recursively split into smaller and smaller subsets
    referred to as nodes, where each newly created node (child) is a subsample
    of a previously existing node (parent). At each node, the target is predicted with
    the average of the target observations in that node.
    </p>

    <p>
    The splits used for creating the nodes are determined through an optimization process
    which minimizes a given loss function under a number of constraints, such as
    that each node should contain a minimum number of samples.
    A common loss function is the mean squared error of the node's prediction,
    which is equal to the variance of the target values in the node, and which therefore
    tends to result in nodes containing similar target values.
    </p>

    <p>
    As the tree grows, finer and finer partitions of the feature space are created,
    reducing the error of the predictions. This process continues until a
    pre-specified condition is met, such as that all terminal nodes, referred to as
    leaves, contain at least a minimum number of observations, or that the depth of
    the tree, as determined by the number of nodes or recursive splits
    from the root node to the leaves, has reached a maximum value.
    </p>

    <p>
    Decision trees are prone to overfitting. A deep enough tree can potentially isolate
    each target value in one leaf, in which case the model predictions exactly match
    the target values observed during training, but are unlikely to provide a good
    approximation for new unseen data that was not used for training. Decision trees
    are also not very robust to the input data, as small changes in the training data
    can potentially result in completely different tree structures.
    </p>

    <img
        id="inflation-forecasting-random-forest-diagram"
        class="blog-post-image"
        style="width:80%"
        alt="Schematic representation of random forest algorithm"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inflation-forecasting-random-forest/diagram_light.png
    />

    <p class="blog-post-image-caption">Schematic representation of random forest algorithm, adapted from
    <a href="#references">[4]</a>.</p>

    <p>
    Random forests address these limitations by creating an ensemble of decision trees
    which are trained on different random subsets of the training data (sample bagging) using
    different random subsets of features (features bagging). The random forest predictions
    are then obtained by averaging the individual predictions of the trees in the ensemble.
    The mechanisms of sample bagging and feature bagging reduce the correlation between
    the predictions of the different trees, making the overall ensemble more robust
    and less prone to overfitting <a href="#references">[3]</a>.
    </p>

******************************************
Data
******************************************
.. raw:: html

    <p>
    As discussed in <a href=https://fg-research.com/blog/general/posts/fred-md-overview.html
    target="_blank">our previous post</a>, FRED-MD is a large, open-source, dataset
    of monthly U.S. macroeconomic indicators maintained by the Federal Reserve Bank of St. Louis.
    The FRED-MD dataset is updated on a monthly basis.
    The monthly releases are referred to as vintages. Each vintage includes the data
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
    We use all vintages from 2025-01 to 2024-02 for our analysis, which is a
    real-time forecasting exercise. On each month, we train the model using
    the data in the vintage released on that month, and generate the forecast
    for the next month. We then compare the forecast to the data in the vintage
    released on the subsequent month.
    </p>

    <p>
    As in <a href="#references">[2]</a>, we include among the features the first 4 principal
    components, which are estimated on all the time series, and the first 4 lags
    of all the time series, including the lags of the principal components and
    the lags of the target time series. This results in approximately 500 features,
    even though the exact number of features changes over time,
    depending on how many time series are included in each vintage.
    </p>

    <img
        id="inflation-forecasting-random-forest-time-series"
        class="blog-post-image"
        style="width:80%"
        alt="US CPI index and corresponding month-over-month logarithmic change"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inflation-forecasting-random-forest/time_series_light.png
    />

    <p class="blog-post-image-caption">US CPI index (FRED: CPIAUCSL) and corresponding month-over-month
    logarithmic change. Source: FRED-MD dataset, 02-2024 vintage.</p>

******************************************
Code
******************************************
We start by importing the dependencies.

.. code:: python

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error
    from scipy.stats import median_abs_deviation

.. raw:: html

    <p>
    After that, we define a number of auxiliary functions for downloading and processing the FRED-MD dataset.
    As discussed in <a href=https://fg-research.com/blog/general/posts/fred-md-overview.html target="_blank">
    our previous post</a>, the FRED-MD dataset includes a set of transformations to be applied to the time
    series in order to ensure their stationarity, which are implemented in the function below.
    </p>

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
            # no transformation
            return x

        elif tcode == 2:
            # first order absolute difference
            return x.diff()

        elif tcode == 3:
            # second order absolute difference
            return x.diff().diff()

        elif tcode == 4:
            # logarithm
            return np.log(x)

        elif tcode == 5:
            # first order logarithmic difference
            return np.log(x).diff()

        elif tcode == 6:
            # second order logarithmic difference
            return np.log(x).diff().diff()

        elif tcode == 7:
            # first order relative difference
            return x.pct_change()

        else:
            raise ValueError(f"unknown `tcode` {tcode}")

.. raw:: html

    <p>
    We then define a function for downloading and processing the data.
    In this function, we download the FRED-MD dataset for the considered vintage,
    transform the time series using the provided transformation codes (with the
    exception of the target time series, for which we use the first order
    logarithmic difference), derive the principal components, and take the
    lags of all the time series.
    </p>

.. code:: python

    def get_data(date, target_name, target_tcode, n_lags, n_components):
        '''
        Download and process the data.

        Parameters:
        ______________________________________________________________
        date: pandas.Timestamp.
            The date of the dataset vintage.

        target_name: string.
            The name of the target time series.

        target_tcode: int.
            The transformation code of the target time series.

        n_lags: int.
            The number of autoregressive lags.

        n_components: int.
            The number of principal components.

        Returns:
        ______________________________________________________________
        train_data: pandas.DataFrame.
            The training dataset.

        test_data: pandas.DataFrame.
            The inputs to the one-month-ahead forecasts.
        '''

        # get the dataset URL
        file = f"https://files.stlouisfed.org/files/htdocs/fred-md/monthly/{date.year}-{format(date.month, '02d')}.csv"

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

        # select the data after January 1960
        data = data[data.index >= pd.Timestamp("1960-01-01")]

        # drop the incomplete time series
        data = data.loc[:, data.isna().sum() == 0]

        # add the principal components
        pca = Pipeline([("scaling", StandardScaler()), ("decomposition", PCA(n_components=n_components))])
        data[[f"PC{i}" for i in range(1, 1 + n_components)]] = pca.fit_transform(data)

        # extract the training data; this includes the target time series and the lags of
        # all time series; the missing values resulting from taking the lags are dropped
        train_data = data[[target_name]].join(data.shift(periods=list(range(1, 1 + n_lags)), suffix="_LAG"))
        train_data = train_data.iloc[n_lags:, :]

        # extract the test data; this includes the last `n_lags` values (e.g. the last 4
        # values) of all time series; the time index is shifted forward by one month to
        # match the date for which the forecasts are generated
        test_data = data.shift(periods=list(range(0, n_lags)), suffix="_LAG")
        test_data = test_data.iloc[-1:, :]
        test_data.index += pd.offsets.MonthBegin(1)
        test_data.columns = [c.split("_LAG_")[0] + "_LAG_" + str(int(c.split("_LAG_")[1]) + 1) for c in test_data.columns]

        return train_data, test_data


.. raw:: html

    <p>
    We also define a function for downloading and processing the target time series.
    We will use this function for obtaining the realized target values against
    which we will compare the forecasts.
    </p>

.. code:: python

    def get_target(start_date, end_date, target_name, target_tcode):
        '''
        Extract the target time series from a range of dataset vintages.

        Parameters:
        ______________________________________________________________
        start_date: pandas.Timestamp.
            The date of the first vintage.

        end_date: pandas.Timestamp.
            The date of the last vintage.

        target_name: str.
            The name of the target time series.

        target_tcode: int.
            The transformation code of the target time series.

        Returns:
        ______________________________________________________________
        targets: pandas.DataFrame.
            The target time series between the start and end date.
        '''

        # create a list for storing the target values
        targets = []

        # loop across the dataset vintages
        for date in tqdm(pd.date_range(start=start_date, end=end_date, freq="MS")):

            # get the dataset URL
            file = f"https://files.stlouisfed.org/files/htdocs/fred-md/monthly/{date.year}-{format(date.month, '02d')}.csv"

            # get the time series
            data = pd.read_csv(file, skiprows=[1], index_col=0)
            data.columns = [c.upper() for c in data.columns]

            # process the dates
            data = data.loc[pd.notna(data.index), :]
            data.index = pd.date_range(start="1959-01-01", freq="MS", periods=len(data))

            # select the target time series
            data = data[[target_name]]

            # transform the target time series
            data[target_name] = transform_series(data[target_name], target_tcode)

            # select the last value
            targets.append(data.iloc[-1:])

        # concatenate the target values in a data frame
        targets = pd.concat(targets, axis=0)

        return targets

.. raw:: html

    <p>
    Finally, we define a function for training the random forest model
    and generating the one-month-ahead forecasts.
    </p>

.. code:: python

    def run_random_forest_model(params, train_data, test_data, target_name):
        '''
        Run the random forest model.

        Parameters:
        ______________________________________________________________
        params: dict.
            The random forest hyperparameters.

        train_data: pandas.DataFrame.
            The training dataset.

        test_data: pandas.DataFrame.
            The inputs to the one-month-ahead forecasts.

        target_name: str.
            The name of the target time series.

        Returns:
        ______________________________________________________________
        forecasts: pandas.Series.
            The one-month-ahead forecasts.
        '''

        # instantiate the model
        model = RandomForestRegressor(**params)

        # fit the model
        model.fit(
            X=train_data.drop(labels=[target_name], axis=1),
            y=train_data[target_name]
        )

        # generate the forecasts
        forecasts = pd.Series(
            data=model.predict(X=test_data),
            index=test_data.index
        )

        return forecasts
.. raw:: html

    <p>
    We define a similar function for the AR model, which we will use as a benchmark.
    </p>

.. code:: python

    def run_autoregressive_model(n_lags, train_data, test_data, target_name):
        '''
        Run the autoregressive model.

        Parameters:
        ______________________________________________________________
        n_lags: int.
            The number of autoregressive lags.

        train_data: pandas.DataFrame.
            The training dataset.

        test_data: pandas.DataFrame.
            The inputs to the one-month-ahead forecasts.

        target_name: str.
            The name of the target time series.

        Returns:
        ______________________________________________________________
        forecasts: pandas.Series.
            The one-month-ahead forecasts.
        '''

        # instantiate the model
        model = LinearRegression(fit_intercept=True)

        # fit the model
        model.fit(
            X=train_data[[f"{target_name}_LAG_{i}" for i in range(1, n_lags + 1)]],
            y=train_data[target_name]
        )

        # generate the forecasts
        forecasts = pd.Series(
            data=model.predict(X=test_data[[f"{target_name}_LAG_{i}" for i in range(1, n_lags + 1)]]),
            index=test_data.index
        )

        return forecasts

.. raw:: html

    <p>
    Lastly, we define a function for iterating over the dataset vintages,
    downloading and processing the data, fitting the random forest and AR models to the data,
    and generating the one-month-ahead forecasts. For comparison purposes, we also include
    the random walk (RW) model, which always predicts that next month's inflation will
    be the same as the current month's inflation.
    </p>

.. code:: python

    def get_forecasts(params, start_date, end_date, target_name, target_tcode, n_lags, n_components):
        '''
        Generate the forecasts over a range of dataset vintages.

        Parameters:
        ______________________________________________________________
        params: dict.
            The random forest hyperparameters.

        start_date: pandas.Timestamp.
            The date of the first vintage.

        end_date: pandas.Timestamp.
            The date of the last vintage.

        target_name: str.
            The name of the target time series.

        target_tcode: int.
            The transformation code of the target time series.

        n_lags: int.
            The number of autoregressive lags.

        n_components: int.
            The number of principal components.

        Returns:
        ______________________________________________________________
        forecasts: pandas.DataFrame.
            The forecasts between the start and end date.
        '''

        # create a list for storing the forecasts
        forecasts = []

        # loop across the dataset vintages
        for date in tqdm(pd.date_range(start=start_date, end=end_date, freq="MS")):

            # get the data
            train_data, test_data = get_data(date, target_name, target_tcode, n_lags, n_components)

            # generate the forecasts
            forecasts.append(pd.DataFrame({
                "RF": run_random_forest_model(params, train_data, test_data, target_name),
                "AR": run_autoregressive_model(n_lags, train_data, test_data, target_name),
                "RW": train_data[target_name].iloc[-1].item()
            }))

        # concatenate the forecasts in a data frame
        forecasts = pd.concat(forecasts, axis=0)

        return forecasts

.. raw:: html

    <p>
    We are now ready to run the analysis.
    We start by defining the target name, which is the FRED name of the US CPI index (CPIAUCSL),
    the target transformation code, which is 5 for first order logarithmic difference, and the dates
    of the first and last vintages used for the analysis.
    </p>

.. code:: python

    target_name = "CPIAUCSL"
    target_tcode = 5
    start_date = pd.Timestamp("2015-01-01")
    end_date = pd.Timestamp("2024-01-01")

.. raw:: html

    <p>
    After that, we generate the one-month-ahead forecasts over the considered time window.
    For the random forest model, we set the number of trees in the ensemble equal to 500, the maximum fraction of
    randomly selected features equal to 1 / 3, and the minimum number of samples in a
    terminal node or leaf equal to 5, as in <a href="#references">[2]</a>. For the autoregressive model,
    we use the same number of lags used by the random forest model which, as discussed above, is equal to 4.
    </p>

.. code:: python

    forecasts = get_forecasts(
            params={
                "n_estimators": 500,
                "max_features": 1 / 3,
                "min_samples_leaf": 5,
                "random_state": 42,
                "n_jobs": -1
            },
            start_date=start_date,
            end_date=end_date,
            target_name=target_name,
            target_tcode=target_tcode,
            n_lags=4,
            n_components=4
        )

.. code:: python

    forecasts.head(n=3)

.. raw:: html

    <img
        id="inflation-forecasting-random-forest-forecasts-table-head"
        class="blog-post-image"
        style="width:80%"
        alt="First 3 values of inflation forecasts"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inflation-forecasting-random-forest/forecasts_table_head_light.png
    />

.. code:: python

    forecasts.tail(n=3)

.. raw:: html

    <img
        id="inflation-forecasting-random-forest-forecasts-table-tail"
        class="blog-post-image"
        style="width:80%"
        alt="Last 3 values of inflation forecasts"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inflation-forecasting-random-forest/forecasts_table_tail_light.png
    />

.. raw:: html

    <p>
    We now download the realized target values.
    </p>

.. code:: python

    targets = get_target(
        start_date=start_date + pd.offsets.MonthBegin(1),
        end_date=end_date + pd.offsets.MonthBegin(1),
        target_name=target_name,
        target_tcode=target_tcode,
    )


.. code:: python

    targets.head(n=3)

.. raw:: html

    <img
        id="inflation-forecasting-random-forest-targets-table-head"
        class="blog-post-image"
        style="width:80%"
        alt="First 3 values of realized inflation"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inflation-forecasting-random-forest/targets_table_head_light.png
    />

.. code:: python

    targets.tail(n=3)

.. raw:: html

    <img
        id="inflation-forecasting-random-forest-targets-table-tail"
        class="blog-post-image"
        style="width:80%"
        alt="Last 3 values of realized inflation"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inflation-forecasting-random-forest/targets_table_tail_light.png
    />

.. raw:: html

    <p>
    Lastly, we calculate the forecast error. We use the root mean squared error (RMSE),
    the mean absolute error (MAE) and the median absolute deviation (MAD) as measures
    of forecast error.
    </p>

.. code:: python

    errors = pd.DataFrame()
    for model in forecasts.columns:
        errors[model] = [
                root_mean_squared_error(y_true=targets[target_name], y_pred=forecasts[model]),
                mean_absolute_error(y_true=targets[target_name], y_pred=forecasts[model]),
                median_abs_deviation(x=targets[target_name] - forecasts[model])
            ]
    errors.index = ["RMSE", "MAE", "MAD"]

.. raw:: html

    <img
        id="inflation-forecasting-random-forest-errors-table"
        class="blog-post-image"
        style="width:80%"
        alt="Forecast errors"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inflation-forecasting-random-forest/errors_table_light.png
    />

.. raw:: html

    <p>
    We find that the random forest model outperforms both the AR model and the RW model
    in terms of all considered error metrics.
    </p>

.. raw:: html

    <img
        id="inflation-forecasting-random-forest-forecasts-plot"
        class="blog-post-image"
        style="width:80%"
        alt="Month-over-month logarithmic change in the US CPI index with random forest (RF) and AR(1) forecasts"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inflation-forecasting-random-forest/forecasts_plot_light.png
    />

    <p class="blog-post-image-caption">Month-over-month logarithmic change in the US CPI index (FRED: CPIAUCSL)
    with random forest (RF) forecasts.</p>


    <p>
    A Python notebook with the full code is available in our
    <a href="https://github.com/fg-research/blog/blob/master/inflation-forecasting-random-forest" target="_blank">GitHub</a>
    repository.
    The official R code from the authors of <a href="#references">[2]</a> is also available in
    <a href="https://github.com/gabrielrvsc/ForecastingInflation" target="_blank">GitHub</a>.
    </p>

******************************************
References
******************************************

[1] McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for macroeconomic research. *Journal of Business & Economic Statistics*, 34(4), 574-589. `doi: 10.1080/07350015.2015.1086655 <https://doi.org/10.1080/07350015.2015.1086655>`__.

[2] Medeiros, M. C., Vasconcelos, G. F., Veiga, Á., & Zilberman, E. (2021). Forecasting inflation in a data-rich environment: the benefits of machine learning methods. *Journal of Business & Economic Statistics*, 39(1), 98-119. `doi: 10.1080/07350015.2019.1637745 <https://doi.org/10.1080/07350015.2019.1637745>`__.

[3] Breiman, L. (2001). Random forests. *Machine learning*, 45, 5-32. `doi: 10.1023/A:101093340432 <https://doi.org/10.1023/A:1010933404324>`__.

[4] Janosh Riebesell. (2022). janosh/tikz: v0.1.0 (v0.1.0). Zenodo. `doi: 10.5281/zenodo.7486911 <https://doi.org/10.5281/zenodo.7486911>`__.