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
    forecasts from February 2023 to January 2024. We find that the random forest model
    outperforms the AR(1) model by almost 20% in terms of root mean squared error (RMSE)
    of the forecasts.
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
    It follows that a decision tree can be seen as a nonparametric regression model,
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
    that each node should contain a certain minimum number of samples.
    An example of loss function is the mean squared error of the node's prediction,
    which is equal to the variance of the target values in the node, and which therefore
    tends to result in nodes containing similar target values.
    </p>

    <p>
    As the tree grows, finer and finer partitions of the feature space are created,
    reducing the error of the predictions. This process continues until a
    pre-specified condition is met, such as that all terminal nodes, referred to as
    leaves, contain at least a given number of observations, or that the depth of
    the tree, as determined by the number of nodes or recursive splits
    from the root node to the leaves, has reached a certain maximum value.
    </p>

    <p>
    Decision trees are prone to overfitting. A deep-enough tree can potentially isolate
    each target value in one leaf, in which case the model predictions exactly match
    the target values observed during training, but are unlikely to provide a good
    approximation for new unseen data that was not used for training. Decision trees
    are also not very robust to the input data, as small changes in the training data
    can potentially result in completely different tree structures.
    </p>

    <p>
    Random forests address these limitations by creating an ensemble of decision trees
    which are trained on different random subsets of the training data (sample bagging) using
    different random subsets of features (features bagging). The random forest predictions
    are then obtained by averaging the individual predictions of the trees in the ensemble.
    The mechanisms of sample bagging and feature bagging reduce the correlation between
    the predictions of the different trees, making the overall ensemble more robust
    and less prone to overfitting <a href="#references">[3]</a>.
    </p>

    <img
        id="inflation-forecasting-random-forest-diagram"
        class="blog-post-image"
        style="width:80%"
        alt="Schematic representation of random forest algorithm"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inflation-forecasting-random-forest/diagram_light.png
    />

    <p class="blog-post-image-caption">Schematic representation of random forest regression algorithm, adapted from
    <a href="#references">[4]</a>.</p>

******************************************
Data
******************************************
.. raw:: html

    <p>
    We use the FRED-MD dataset for developing and validating the random forest model.
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
    We use 02-2023 vintage for training and hyperparameter tuning, while we use the last
    month in each vintage from 03-2023 to 02-2024 for testing. Our approach is different
    from the one used in <a href="#references">[2]</a>, where the same vintage (01-2016)
    is used for both training and testing. In our view, our approach allows us to evaluate
    the model in a more realistic scenario, as on a given month we forecast next month's
    inflation using as input the data available on that month, without taking into account
    any ex-post adjustment that could be applied to the data in the future.
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
In this section, we provide and explain the Python code used for the analysis.

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

    <p>
    We then define a function for downloading and processing the training data.
    In this function, we download the FRED-MD dataset for the considered vintage,
    transform the time series using the provided transformation codes (with the
    exception of the target time series, for which we use the first order
    logarithmic difference as in <a href="#references">[2]</a>) and define the
    features as the first lag (i.e. the previous month value) of all the time series
    (including the target time series). As in <a href="#references">[2]</a>,
    we use the data after January 1960, and we use only the time series without
    missing values.
    </p>

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

        # move the target to the first column
        data = data[[target_name] + data.columns.drop(target_name).tolist()]

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

        # select the time series without missing values
        data = data.loc[:, data.isna().sum() == 0]

        # lag the features
        data = data[[target_name]].join(data.shift(periods=[1], suffix="_LAG"))

        # drop the missing value resulting from taking the lag
        return data.iloc[1:]

.. raw:: html

    <p>
    For the test data, we download and process the targets and features separately,
    given that they are extracted from different vintages. The targets are extracted
    from the vintages between 03-2023 and 02-2024, while the features are extracted
    from the vintages between 02-2023 and 01-2024.
    </p>

    <p>
    The following function extracts the target values. It iterates across the selected
    vintages, downloads the data for each vintage, transforms the target time series,
    and returns its last value, i.e. its value on the last month of each vintage.
    </p>

.. code:: python

    def get_target(start_month, start_year, end_month, end_year, target_name, target_tcode):
        '''
        Extract the target time series from a range of dataset vintages.

        Parameters:
        ______________________________________________________________
        start_month: int.
            The month of the first vintage.

        start_year: int.
            The year of the first vintage.

        end_month: int.
            The month of the last vintage.

        end_year: int.
            The year of the last vintage.

        target_name: str.
            The name of the target time series.

        target_tcode: int.
            The transformation code of the target time series.
        '''

        # create a data frame for storing the target values
        target = pd.DataFrame()

        # define the date range of the dataset vintages
        dates = pd.date_range(
            start=f"{start_year}-{start_month}-01",
            end=f"{end_year}-{end_month}-01",
            freq="MS"
        )

        # loop across the dataset vintages
        for date in dates:

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

            # select the last value and save it in the data frame
            target = pd.concat([target, data.iloc[-1:]], axis=0)

        return target

.. raw:: html

    <p>
    The following function extracts the feature values. Given that the model
    uses the first lag of the features, i.e. their values on the previous month,
    we shift back the dates of the dataset vintages by one month. We then
    download the time series in each vintage, transform the time series,
    and return their last values, i.e. their values on the last month of each vintage.
    After that we shift the dates forward by one month, such that we can correctly map
    the feature values observed on a given month to the corresponding target values
    observed in the subsequent month.
    </p>

.. code:: python

    def get_features(start_month, start_year, end_month, end_year, target_name, target_tcode, feature_names):
        '''
        Extract the feature time series from a range of dataset vintages.

        Parameters:
        ______________________________________________________________
        start_month: int.
            The month of the first vintage.

        start_year: int.
            The year of the first vintage.

        end_month: int.
            The month of the last vintage.

        end_year: int.
            The year of the last vintage.

        target_name: str.
            The name of the target time series.

        target_tcode: int.
            The transformation code of the target time series.

        feature_names: list of str.
            The names of the features time series.
        '''

        # create a data frame for storing the feature values
        features = pd.DataFrame()

        # define the date range of the dataset vintages
        dates = pd.date_range(
            start=f"{start_year}-{start_month}-01",
            end=f"{end_year}-{end_month}-01",
            freq="MS"
        )

        # loop across the dataset vintages
        for date in dates:

            # get the dataset URL
            file = f"https://files.stlouisfed.org/files/htdocs/fred-md/monthly/{(date - pd.offsets.MonthBegin(1)).year}-{format((date - pd.offsets.MonthBegin(1)).month, '02d')}.csv"

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

            # rename the time series
            data.columns = [c + "_LAG_1" for c in data.columns]

            # drop any features that were not used for training
            data = data[feature_names]

            # forward fill any missing values
            data = data.ffill()

            # shift the dates one month forward
            data.index += pd.offsets.MonthBegin(1)

            # select the last values and save them in the data frame
            features = pd.concat([features, data.iloc[-1:]], axis=0)

        return features

.. raw:: html

    <p>
    The function below extract the target and features from the different
    dataset vintages as outlined above, and merges them into a unique data frame.
    </p>

.. code:: python

    def get_test_data(start_month, start_year, end_month, end_year, target_name, target_tcode, feature_names):
        '''
        Download and process the test data.

        Parameters:
        ______________________________________________________________
        start_month: int.
            The month of the first vintage.

        start_year: int.
            The year of the first vintage.

        end_month: int.
            The month of the last vintage.

        end_year: int.
            The year of the last vintage.

        target_name: str.
            The name of the target time series.

        target_tcode: int.
            The transformation code of the target time series.

        feature_names: list of str.
            The names of the features time series.
        '''

        # get the targets
        target = get_target(
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            end_month=end_month,
            target_name=target_name,
            target_tcode=target_tcode,
        )

        # get the features
        features = get_features(
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            end_month=end_month,
            target_name=target_name,
            target_tcode=target_tcode,
            feature_names=feature_names
        )

        return target.join(features)

.. raw:: html

    <br>
    Finally, we define a function for training the random forest model
    and generating the test set predictions.
    <br>

.. code:: python

    def run_random_forest_model(params, train_dataset, test_dataset, target_name):
        '''
        Run the random forest model.

        Parameters:
        ______________________________________________________________
        params: dict.
            The random forest hyperparameters

        train_dataset: pandas.DataFrame.
            Training dataset.

        test_dataset: pandas.DataFrame.
            Test dataset.

        target_name: str.
            The name of the target time series.
        '''

        # instantiate the model
        model = RandomForestRegressor(**params)

        # fit the model to the training set
        model.fit(
            X=train_dataset.drop([target_name], axis=1),
            y=train_dataset[target_name]
        )

        # generate the forecasts over the test set
        return pd.Series(
            data=model.predict(X=test_dataset.drop([target_name], axis=1)),
            index=test_dataset.index
        )

.. raw:: html

    <br>
    We also define a similar function for the AR(1) model, which we will use as a benchmark.
    <br>

.. code:: python

    def run_autoregressive_model(train_dataset, test_dataset, target_name):
        '''
        Run the AR(1) model.

        Parameters:
        ______________________________________________________________
        train_dataset: pandas.DataFrame.
            Training dataset.

        test_dataset: pandas.DataFrame.
            Validation dataset.

        target_name: str.
            The name of the target time series.
        '''

        # instantiate the model
        model = LinearRegression(fit_intercept=True)

        # fit the model to the training set
        model.fit(
            X=train_dataset[[target_name + "_LAG_1"]],
            y=train_dataset[target_name]
        )

        # generate the forecasts over the test set
        return pd.Series(
            data=model.predict(X=test_dataset[[target_name + "_LAG_1"]]),
            index=test_dataset.index
        )

Lastly, we define an additional function which uses `optuna <https://optuna.org/>`__
for tuning the main hyperparameters of the random forest model.

.. code:: python

    def tune_random_forest_model(train_dataset, valid_dataset, target_name, n_trials):
        '''
        Tune the random forest hyperparameters.

        Parameters:
        ______________________________________________________________
        train_dataset: pandas.DataFrame.
            Training dataset.

        valid_dataset: pandas.DataFrame.
            Validation dataset.

        target_name: str.
            The name of the target time series.

        n_trials: int.
            The number of random search iterations.
        '''

        # define the objective function
        def objective(trial):

            # sample the hyperparameters
            params = {
                "criterion": trial.suggest_categorical("criterion", choices=["absolute_error", "squared_error"]),
                "n_estimators": trial.suggest_int("n_estimators", low=10, high=100),
                "max_features": trial.suggest_float("max_features", low=0.6, high=1.0),
                "max_samples": trial.suggest_float("max_samples", low=0.6, high=1.0),
                "max_depth": trial.suggest_int("max_depth", low=1, high=100),
                "random_state": trial.suggest_categorical("random_state", choices=[42]),
                "n_jobs": trial.suggest_categorical("n_jobs", choices=[-1])
            }

            # calculate the root mean squared error of the forecasts
            return root_mean_squared_error(
                y_true=valid_dataset[target_name],
                y_pred=run_random_forest_model(
                    params=params,
                    train_dataset=train_dataset,
                    test_dataset=valid_dataset,
                    target_name=target_name
                )
            )

        # minimize the objective function
        study = optuna.create_study(
            sampler=optuna.samplers.RandomSampler(seed=42),
            direction="minimize"
        )

        study.optimize(
            func=objective,
            n_trials=n_trials
        )

        # return the best hyperparameters
        return study.best_params

==========================================
Analysis
==========================================
We are now ready to run the analysis.
We start by defining the target name, which is the FRED name of the US CPI index ("CPIAUCSL"),
and the target transformation code, which is 5 for first order logarithmic difference.

.. code:: python

    target_name = "CPIAUCSL"
    target_tcode = 5

After that we load the training dataset, which contains 756 monthly observations
on 110 variables. The variables include the target time series, the first lag of the target
time series, and the first lag of 108 macroeconomic indicators with complete
time series (i.e. without missing values) from February 1960 to January 2023.

.. code:: python

    train_dataset = get_training_data(
        year=2023,
        month=2,
        target_name=target_name,
        target_tcode=target_tcode
    )

We then proceed to tuning the random forest hyperparameters
by performing random search with `optuna <https://optuna.org/>`__. We use
the last 12 months of the training set as validation set
and we use the RMSE as objective function.

.. code:: python

    params = tune_random_forest_model(
        train_dataset=train_dataset.iloc[:-12],
        valid_dataset=train_dataset.iloc[-12:],
        target_name=target_name,
        n_trials=30
    )

The identified best hyperparameters are reported below.

.. code:: python

    {
        'criterion': 'absolute_error',
        'n_estimators': 12,
        'max_features': 0.6431565707973218,
        'max_samples': 0.6125716742746937,
        'max_depth': 64
    }

We now load the test dataset, which contains 12 monthly observations
on the same 110 variables from February 2023 to January 2024.

.. code:: python

    test_dataset = get_test_data(
        start_year=2023,
        start_month=3,
        end_year=2024,
        end_month=2,
        target_name=target_name,
        target_tcode=target_tcode,
        feature_names=train_dataset.columns.drop(target_name).tolist()
    )

We can finally train the random forest model using the identified best
hyperparameters, generate the forecasts over the test set, and
calculate the RMSE of the forecasts.

.. code:: python

    rf_forecasts = run_random_forest_model(
        params=params,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        target_name=target_name
    )

    rf_error = root_mean_squared_error(
        y_true=forecasts[target_name],
        y_pred=rf_forecasts
    )


We do the same for the AR(1) model.

.. code:: python

    ar1_forecasts = run_autoregressive_model(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        target_name=target_name,
    )

    ar1_error = root_mean_squared_error(
        y_true=forecasts[target_name],
        y_pred=ar1_forecasts
    )

The RMSE of the random forest model is 0.001649,
while the RMSE of the AR(1) model is 0.002023.
The reduction in RMSE provided by the random forest model is 18.5%.

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

.. tip::

    A Python notebook with the full code is available in our
    `GitHub repository <https://github.com/fg-research/blog/blob/master/inflation-forecasting-random-forest/inflation-forecasting-random-forest.ipynb>`__.

******************************************
References
******************************************

[1] McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for macroeconomic research. *Journal of Business & Economic Statistics*, 34(4), 574-589. `doi: 10.1080/07350015.2015.1086655 <https://doi.org/10.1080/07350015.2015.1086655>`__.

[2] Medeiros, M. C., Vasconcelos, G. F., Veiga, Á., & Zilberman, E. (2021). Forecasting inflation in a data-rich environment: the benefits of machine learning methods. *Journal of Business & Economic Statistics*, 39(1), 98-119. `doi: 10.1080/07350015.2019.1637745 <https://doi.org/10.1080/07350015.2019.1637745>`__.

[3] Breiman, L. (2001). Random forests. *Machine learning*, 45, 5-32. `doi: 10.1023/A:101093340432 <https://doi.org/10.1023/A:1010933404324>`__.

[4] Janosh Riebesell. (2022). janosh/tikz: v0.1.0 (v0.1.0). Zenodo. `doi: 10.5281/zenodo.7486911 <https://doi.org/10.5281/zenodo.7486911>`__.