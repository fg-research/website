.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Predicting stock market trends with Amazon SageMaker Autopilot
   :keywords: Amazon SageMaker, Time Series, AutoML, Forecasting, Stock Market

######################################################################################
Predicting stock market trends with Amazon SageMaker Autopilot
######################################################################################

.. raw:: html

    <p>
    Building a well-performing machine learning model requires substantial time and resources.
    Automated machine learning (AutoML) automates the end-to-end process of building, training
    and tuning machine learning models.
    This not only accelerates the development cycle, but also makes machine learning more accessible
    to those without specialized data science expertise.
    </p>

    <p>
    In this post, we use <a href=https://aws.amazon.com/sagemaker/autopilot target=_blank>
    SageMaker Autopilot </a> for building a stock market trend prediction model.
    We will use AutoML for building an ensemble of gradient boosting classifiers
    (XGBoost, LightGBM and CatBoost) to predict the direction of the S&P 500 (up or down)
    one day ahead using as input a set of technical indicators.
    </p>

    <p>
    We will download the S&P 500 daily time series from the
    1<sup>st</sup> of August 2021 to the 31<sup>st</sup> of July 2024 from
    <a href="https://finance.yahoo.com" target="_blank">Yahoo! Finance</a>.
    We will train the model on the data up to the 3<sup>rd</sup> of May 2024 (training set),
    and validate the model on the subsequent 30 days of data up to the
    17<sup>th</sup> of June 2024 (validation set).
    We will then test the identified best model, i.e. the one with the best performance
    on the validation set, on the remaining 30 days of data up to the
    31<sup>st</sup> of July 2024 (test set).
    We will find that the model achieves a mean directional accuracy of 63%
    over the test set.
    </p>

******************************************
Data
******************************************

==========================================
Outputs
==========================================
The model output is the sign of the next day's price move of the S&P 500,
which is derived as follows

.. math::

    \begin{equation}
      D_{it} =
        \begin{cases}
          1 & \text{if bank $i$ issues ABs at time $t$}\\
          2 & \text{if bank $i$ issues CBs at time $t$}\\
          0 & \text{otherwise}
        \end{cases}
    \end{equation}

==========================================
Inputs
==========================================


******************************************
Code
******************************************

==========================================
Environment Set-Up
==========================================

We start by importing all the dependencies and setting up the SageMaker environment.

.. note::

    We use the :code:`yfinance` library for downloading the S&P500 daily time series and
    the :code:`pyti` library for calculating the technical indicators.

.. code:: python

    import warnings
    import io
    import boto3
    import json
    import sagemaker
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from pyti.simple_moving_average import simple_moving_average
    from pyti.weighted_moving_average import weighted_moving_average
    from pyti.momentum import momentum
    from pyti.stochastic import percent_k, percent_d
    from pyti.williams_percent_r import williams_percent_r
    from pyti.accumulation_distribution import accumulation_distribution
    from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence
    from pyti.relative_strength_index import relative_strength_index
    from pyti.commodity_channel_index import commodity_channel_index
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
    warnings.filterwarnings(action="ignore")

    # SageMaker session
    session = sagemaker.Session()

    # SageMaker role
    role = sagemaker.get_execution_role()

    # S3 bucket
    bucket = session.default_bucket()

    # Boto3 client
    client = boto3.client("sagemaker-runtime")

==========================================
Data Preparation
==========================================

.. raw:: html

    <p>
    Next, we download the S&P 500 time series from the 1<sup>st</sup> of August 2021 to the 31<sup>st</sup> of July 2024.
    The dataset contains 754 daily observations.
    </p>

.. code:: python

    # download the data
    dataset = yf.download(tickers="^SPX", start="2021-08-01", end="2024-08-01")

******************************************
References
******************************************

[1] Kara, Y., Boyacioglu, M. A., & Baykan, Ö. K. (2011).
Predicting direction of stock price index movement using artificial neural networks and support vector machines:
The sample of the Istanbul Stock Exchange. *Expert Systems with Applications*, 38(5), 5311-5319.
`doi: doi:10.1016/j.eswa.2010.10.027 <https://doi.org/doi:10.1016/j.eswa.2010.10.027>`__.

