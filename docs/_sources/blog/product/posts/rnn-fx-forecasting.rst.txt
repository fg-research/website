.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Forecasting Stock Returns with Liquid Neural Networks
   :keywords: Amazon SageMaker, Time Series, Recurrent Neural Networks, Forecasting, Forex Market

############################################################################################################
Forecasting exchange rates with long short-term memory (LSTM) networks using the RNN SageMaker Algorithm
############################################################################################################

.. raw:: html

    <p>
    Forecasting exchange rates is a critical task for traders, investors, and financial institutions.
    Even though different machine learning models have been studied for this purpose, Long Short-Term Memory
    (LSTM) <a href="#references">[1]</a> networks have become the most widely adopted <a href="#references">[2]</a>.
    </p>

    <p>
    LSTMs belong to the class of recurrent neural networks (RNN), which are designed to process and predict
    sequences of data. While vanilla RNNs often fail to capture long-term dependencies due to issues like
    vanishing gradients, LSTMs implement a number of gating mechanisms which allow the network to retain memory
    of relevant features over longer time intervals.
    </p>

    <p>
    In this post, we will use our Amazon SageMaker implementation of RNNs for
    probabilistic time series forecasting, the <a href="https://fg-research.com/algorithms/time-series-forecasting/index.html#rnn-sagemaker-algorithm" target="_blank">RNN SageMaker algorithm</a>,
    for implementing an LSTM model with two layers with LeCun's hyperbolic tangent activation <a href="#references">[3]</a>.
    We will use the model for generating one-day-ahead forecasts of the EUR/USD exchange rate using as input a
    set of technical indicators, similar to <a href="#references">[4]</a>.
    </p>

    <p>
    We will use the daily EUR/USD exchange rate from the 1<sup>st</sup> of August 2022 to
    the 31<sup>st</sup> of July 2024, which we will download from
    <a href="https://finance.yahoo.com" target="_blank">Yahoo! Finance</a>.
    We will train the model on the data up to the 18<sup>th</sup> of June 2024,
    and use the trained model to predict the subsequent data up to the 31<sup>st</sup> of July 2024.
    We will find that our LSTM model achieves a mean absolute error of 0.0012 and
    a mean directional accuracy of 83.33% over the considered time period.
    </p>

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

    # SageMaker algorithm ARN, replace the placeholder below with your AWS Marketplace ARN
    algo_arn = "arn:aws:sagemaker:<...>"

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

==========================================
Data Preparation
==========================================

.. raw:: html

    <p>
    Next, we download the EUR/USD time series from the 1<sup>st</sup> of August 2022 to
    the 31<sup>st</sup> of July 2024 using the <a href="https://github.com/ranaroussi/yfinance" target="_blank">Yahoo! Finance Python API</a>.
    The dataset contains 522 daily observations.
    </p>

.. code:: python

    dataset = yf.download(tickers="EURUSD=X", start="2022-08-01", end="2024-08-01")

.. raw:: html

    <p>
    We then calculate the following technical indicators, as in <a href="#references">[2]</a>:
    </p>

* Moving average (MA) with a period of 10.

* Moving average convergence/divergence (MACD) with short- and long-term periods of 12 and 26.

* Rate of change (ROC) with a period of 2.

* Momentum with a period of 4.

* Relative strength index (RSI) with a period of 10.

* Bollinger bands (BB) with period of 20.

* Commodity channel index (CCI) with a period of 20.

.. code:: python

    # MA with a period of 10
    dataset["MA"] = simple_moving_average(
        data=dataset["Close"],
        period=10
    )

    # MACD with short- and long-term periods of 12 and 26
    dataset["MACD"] = moving_average_convergence_divergence(
        data=dataset["Close"],
        short_period=12,
        long_period=26
    )

    # ROC with a period of 2
    dataset["ROC"] = rate_of_change(
        data=dataset["Close"],
        period=2
    )

    # Momentum with a period of 4
    dataset["Momentum"] = momentum(
        data=dataset["Close"],
        period=4
    )

    # RSI with a period of 10
    dataset["RSI"] = relative_strength_index(
        data=dataset["Close"],
        period=10
    )

    # BB with period of 20
    dataset["MiddleBB"] = middle_bollinger_band(
        data=dataset["Close"],
        period=20
    )

    dataset["LowerBB"] = upper_bollinger_band(
        data=dataset["Close"],
        period=20
    )

    dataset["UpperBB"] = lower_bollinger_band(
        data=dataset["Close"],
        period=20
    )

    # CCI with a period of 20
    dataset["CCI"] = commodity_channel_index(
        close_data=dataset["Close"],
        low_data=dataset["Low"],
        high_data=dataset["High"],
        period=20
    )

.. code:: python

    # drop the missing values
    dataset.dropna(inplace=True)

After dropping the missing values resulting from the calculation of the technical indicators,
the number of daily observations is reduced to 497.

.. raw:: html

    <img
        id="rnn-fx-forecasting-time-series"
        class="blog-post-image"
        alt="EUR/USD daily exchange rate with technical indicators from 2022-09-05 to 2024-07-31"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/rnn-fx-forecasting/time_series_light.png
    />

    <p class="blog-post-image-caption">EUR/USD daily exchange rate with technical indicators from 2022-09-05 to 2024-07-31.</p>

We now proceed to renaming the columns in the format expected by the RNN SageMaker algorithm,
where the output names should start with :code:`"y"` and the input names should start with :code:`"x"`.

.. code:: python

    # drop the unnecessary columns
    dataset.drop(labels=["Adj Close", "Volume"], axis=1, inplace=True)

    # move the target to the first column
    dataset = dataset[["Close"] + dataset.columns.drop("Close").tolist()]

    # rename the columns
    dataset.columns = ["y"] + [f"x{i}" for i in range(dataset.shape[1] - 1)]

.. note::

    Note that the algorithm's code always includes the past values of the outputs
    among the inputs and, therefore, there is no need to add the lagged values of
    the outputs when preparing the data for the model.

We then split the data into a training set and a test set.
We use the last 30 days for testing, and the previous 467 days for training.
We save both the training data and the test data to CSV files in S3.

.. code:: python

    # define the size of the test set
    test_size = 30

    # extract the training data
    training_dataset = dataset.iloc[:- test_size]

    # extract the test data
    test_dataset = dataset.iloc[- test_size - context_length:]

    # upload the training data to S3
    training_data = sagemaker_session.upload_string_as_file_body(
        body=training_dataset.to_csv(index=False),
        bucket=bucket,
        key="training_data.csv"
    )

    # upload the test data to S3
    test_data = sagemaker_session.upload_string_as_file_body(
        body=test_dataset.to_csv(index=False),
        bucket=bucket,
        key="test_data.csv"
    )

.. note::

    Note that the data is scaled internally by the algorithm, there is no need to scale the data beforehand.

==========================================
Training
==========================================
We can now train the model using the data in S3.
We use two LSTM layers with respectively 100 and 50 hidden units and apply a LeCun's hyperbolic tangent activation after each layer.
We train the model for 200 epochs with a batch size of 16 and a learning rate of 0.001, where the learning rate is decayed exponentially at a rate of 0.99.

.. code:: python

    # create the estimator
    estimator = sagemaker.algorithm.AlgorithmEstimator(
        algorithm_arn=algo_arn,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        input_mode="File",
        sagemaker_session=sagemaker_session,
        hyperparameters={
            "context-length": context_length,
            "prediction-length": prediction_length,
            "sequence-stride": 1,
            "cell-type": "lstm",
            "hidden-size-1": 100,
            "hidden-size-2": 50,
            "hidden-size-3": 0,
            "activation": "lecun",
            "dropout": 0,
            "batch-size": 16,
            "lr": 0.001,
            "lr-decay": 0.99,
            "epochs": 200,
        }
    )

    # run the training job
    estimator.fit({"training": training_data})

==========================================
Inference
==========================================
After the training job has been completed, we run a batch transform job on the test data in S3.
The results are saved to a CSV file in S3 with the same name as the input CSV file but with the :code:`".out"` file extension.

.. code:: python

    # create the transformer
    transformer = estimator.transformer(
        instance_count=1,
        instance_type=instance_type,
    )

    # run the transform job
    transformer.transform(
        data=test_data,
        content_type="text/csv",
    )

After the batch transform job has been completed, we can load the results from S3.
For the purpose of evaluating the model's directional accuracy, we calculate the
1-day predicted returns, that is the 1-day percentage changes predicted by the model.

.. code:: python

    # get the forecasts from S3
    predictions = sagemaker_session.read_s3_file(
        bucket=bucket,
        key_prefix=f"{transformer.latest_transform_job.name}/test_data.csv.out"
    )

    # cast the forecasts to data frame
    predictions = pd.read_csv(io.StringIO(predictions), dtype=float)

    # drop the out-of-sample forecast
    predictions = predictions.iloc[:-1]

    # add the dates
    predictions.index = test_dataset.index

    # add the actual values
    predictions["y"] = test_dataset["y"]

    # add the actual and predicted percentage changes
    predictions["r"] = predictions["y"] / predictions["y"].shift(periods=1) - 1
    predictions["r_mean"] = predictions["y_mean"] / predictions["y"].shift(periods=1) - 1

    # drop the missing values
    predictions.dropna(inplace=True)

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

==========================================
Evaluation
==========================================
We evaluate the test set predictions using the following metrics:

* The root mean squared error (*RMSE*) of the predicted values.

* The mean absolute error (*MAE*) of the predicted values.

* The *accuracy* of the predicted signs of the returns.

* The *F1* score of the predicted signs of the returns.

.. raw:: html

    <img
        id="rnn-fx-forecasting-metrics"
        class="blog-post-image"
        alt="Performance metrics of predicted EUR/USD daily exchange rate over the test set (from 2024-06-19 to 2024-07-31)"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/rnn-fx-forecasting/metrics_light.png
    />

    <p class="blog-post-image-caption">Performance metrics of predicted EUR/USD daily exchange rate over the test set (from 2024-06-19 to 2024-07-31).</p>

We find that the model achieves a mean absolute error of 0.0012 and a mean directional accuracy of 83.33% on the test set.

We can now delete the model.

.. code:: python

    # delete the model
    transformer.delete_model()

.. tip::

    You can download the
    `notebook <https://github.com/fg-research/rnn-sagemaker/blob/master/examples/EURUSD.ipynb>`__
    with the full code from our
    `GitHub <https://github.com/fg-research/rnn-sagemaker>`__
    repository.

******************************************
References
******************************************

[1] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8), pp. 1735-1780.
`doi: 10.1162/neco.1997.9.8.1735 <https://doi.org/10.1162/neco.1997.9.8.1735>`__.

[2] Ayitey Junior, M., Appiahene, P., Appiah, O., & Bombie, C. N. (2023).
Forex market forecasting using machine learning: systematic literature review and meta-analysis. *Journal of Big Data*, 10(1), 9.
`doi: 10.1186/s40537-022-00676-2 <https://doi.org/10.1186/s40537-022-00676-2>`__.

[3] LeCun, Y., Bottou, L., Orr, G. B., & Müller, K. R. (2002). Efficient backprop. In *Neural networks: Tricks of the trade.*, pp. 9-50, Springer.
`doi: 10.1007/3-540-49430-8_2 <https://doi.org/10.1007/3-540-49430-8_2>`__.

[4] Yıldırım, D. C., Toroslu, I. H., & Fiore, U. (2021). Forecasting directional movement of Forex data using LSTM with technical and macroeconomic indicators.
*Financial Innovation*, 7, pp. 1-36. `doi: 10.1186/s40854-020-00220-2 <https://doi.org/10.1186/s40854-020-00220-2>`__.
