.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Forecasting Stock Returns with Liquid Neural Networks
   :keywords: Amazon SageMaker, Time Series, Recurrent Neural Networks, Forecasting, Forex Market

############################################################################################################
Forecasting exchange rates with long short-term memory (LSTM) networks using the RNN SageMaker Algorithm
############################################################################################################


******************************************
Data
******************************************
The model generates one-day-ahead predictions of the EUR/USD exchange rate using as input the past values of the
EUR/USD exchange rate and of the following technical indicators:



******************************************
Code
******************************************

We start by importing all the dependencies and setting up the SageMaker environment.

.. warning::

   To be able to run the code below, you need to have an active subscription to the
   RNN SageMaker algorithm. You can subscribe to a free trial from the
   `AWS Marketplace <https://aws.amazon.com/marketplace/pp/prodview-p5cr7ncmdcziw>`__
   in order to get your Amazon Resource Name (ARN).
   In this post we use version 1.0 of the RNN SageMaker algorithm, which runs in the
   PyTorch 2.1.0 Python 3.10 deep learning container.

We use the :code:`yfinance` library for downloading the EUR/USD daily time series and
the :code:`pyti` library for calculating the technical indicators.

.. code:: python

    import io
    import sagemaker
    import warnings
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from pyti.momentum import momentum
    from pyti.rate_of_change import rate_of_change
    from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence
    from pyti.simple_moving_average import simple_moving_average
    from pyti.relative_strength_index import relative_strength_index
    from pyti.bollinger_bands import middle_bollinger_band, upper_bollinger_band, lower_bollinger_band
    from pyti.commodity_channel_index import commodity_channel_index
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error, accuracy_score, f1_score
    warnings.filterwarnings(action='ignore')

    # SageMaker session
    sagemaker_session = sagemaker.Session()

    # SageMaker role
    role = sagemaker.get_execution_role()

    # S3 bucket
    bucket = sagemaker_session.default_bucket()

    # EC2 instance
    instance_type = "ml.m5.4xlarge"

After that we define the neural network's *context length* and *prediction length*.
The context length is the number of past time steps used as input,
while the prediction length is the number of future time steps to be predicted.
We set the context length equal to 5 and the prediction length equal to 1, that is
we use the values of the EUR/USD exchange rate and of the technical indicators on
the previous 5 days to predict the value of the EUR/USD exchange rate on the next day.

.. code:: python


    # number of time steps used as input
    context_length = 5

    # number of time steps to output
    prediction_length = 1


.. raw:: html

    <img
        id="rnn-fx-forecasting-time-series"
        class="blog-post-image"
        alt="EUR/USD daily exchange rate with technical indicators from 2022-09-05 to 2024-07-31"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/rnn-fx-forecasting/time_series_light.png
    />

    <p class="blog-post-image-caption">EUR/USD daily exchange rate with technical indicators from 2022-09-05 to 2024-07-31.</p>


.. raw:: html

    <img
        id="rnn-fx-forecasting-predictions"
        class="blog-post-image"
        alt="Actual and predicted EUR/USD daily exchange rate over the test set (from 2024-06-19 to 2024-07-31)."
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/rnn-fx-forecasting/predictions_light.png
    />

    <p class="blog-post-image-caption">Actual and predicted EUR/USD daily exchange rate over the test set (from 2024-06-19 to 2024-07-31).</p>



.. raw:: html

    <img
        id="rnn-fx-forecasting-returns"
        class="blog-post-image"
        alt="Actual and predicted EUR/USD daily percentage changes over the test set (from 2024-06-19 to 2024-07-31)."
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/rnn-fx-forecasting/returns_light.png
    />

    <p class="blog-post-image-caption">Actual and predicted EUR/USD daily percentage changes over the test set (from 2024-06-19 to 2024-07-31).</p>


.. raw:: html

    <img
        id="rnn-fx-forecasting-metrics"
        class="blog-post-image"
        alt="Performance metrics of predicted EUR/USD daily exchange rate over the test set (from 2024-06-19 to 2024-07-31)"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/rnn-fx-forecasting/metrics_light.png
    />

    <p class="blog-post-image-caption">Performance metrics of predicted EUR/USD daily exchange rate over the test set (from 2024-06-19 to 2024-07-31).</p>
