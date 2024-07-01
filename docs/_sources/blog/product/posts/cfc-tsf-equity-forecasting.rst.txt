.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Forecasting Stock Returns with Liquid Neural Networks
   :keywords: Amazon SageMaker, Time Series, Liquid Neural Networks, Forecasting

###########################################################################################
Forecasting stock returns with liquid neural networks using the CfC SageMaker Algorithm
###########################################################################################

.. raw:: html

    <p>
    Stock return forecasting has been extensively studied by both academic researchers and
    industry practitioners. Numerous machine learning models have been proposed for this purpose,
    ranging from simple linear regressions to complex deep learning models <a href="#references">[1]</a>.
    In this post, we examine the performance of liquid neural networks <a href="#references">[4]</a>
    <a href="#references">[5]</a>, a new neural network architecture for sequential data.
    </p>

    <p>
    We will use our Amazon SageMaker implementation of liquid neural networks for probabilistic time series
    forecasting, the <a href="file:///Users/flaviagiammarino/website/docs/algorithms/time-series-forecasting/index.html#cfc-sagemaker-algorithm" target="_blank"> CfC SageMaker algorithm</a>.
    We will forecast the conditional mean and the conditional standard deviation of the 30-day returns of
    the S&P 500 using as input the S&P 500 realized volatility as well as several implied volatility indices,
    similar to <a href="#references">[2]</a>.
    </p>

    <p>
    We will use the daily close prices from the 30<sup>th</sup> of June 2022 to
    the 29<sup>th</sup> of June 2024, which we will download with the <a href="https://github.com/ranaroussi/yfinance" target="_blank">Yahoo! Finance Python API</a>.
    We will train the model on the data up to the 8<sup>th</sup> of September 2023,
    and use the trained model to predict the subsequent data up to the 29<sup>th</sup> of June 2024.
    We will find that the CfC SageMaker algorithm achieves a mean absolute error of 1.4% and
    a mean directional accuracy of 97.5%.
    </p>

******************************************
Model
******************************************

.. raw:: html

    <p>
    The closed-form continuous-depth network (CfC) is a new neural network architecture for
    sequential data <a href="#references">[4]</a>. CfCs belong to the class of continuous-time
    recurrent neural networks (CT-RNNs) <a href="#references">[3]</a>, where the evolution of
    the hidden state over time is described by an Ordinary Differential Equation (ODE).
    </p>

    <p>
    CfCs use the Liquid Time Constant (LTC) ODE <a href="#references">[5]</a>, where both the
    derivative and the time constant of the hidden state are determined by a neural network.
    Differently from other CT-RNNs (including LTCs), which use a numerical solver to find the
    ODE solution, CfCs use an approximate closed-form solution. As a results, CfCs achieve
    faster training and inference performance than other CT-RNNs.
    </p>

The hidden state :math:`x` of a CfC at time :math:`t` is given by

.. math::

    x(t) = \sigma(-f(x, I; \theta_f)t) \odot g(x, I; \theta_g) + [1 - \sigma(-[f(x, I; \theta_f)]t)] \odot h(x, I; \theta_h)

where :math:`\odot` is the Hadamard product, :math:`\sigma` is the sigmoid function, :math:`I`
is the input sequence, while :math:`f`, :math:`g` and :math:`h` are neural networks. The three
neural networks :math:`f`, :math:`g` and :math:`h` share a common backbone, which is a stack of
fully-connected layers with non-linear activation.

The backbone is followed by three separate neural network heads. The head of the :math:`g` and
:math:`h` neural networks is a fully-connected layer with hyperbolic tangent activation. The head
of the :math:`f` neural network is an affine function :math:`b + a(\Delta t)` where :math:`\Delta t`
is the time span (or time increment) between consecutive time steps, while the intercept :math:`b`
and slope :math:`a` are the outputs of two fully-connected layers with linear activation.

******************************************
Data
******************************************

==========================================
Outputs
==========================================

The model outputs are the 30-day returns of the S&P 500, which are calculated as follows

.. math::

    y(t) = \ln{P(t)} - \ln{P(t-30)}

for each day :math:`t`, where :math:`P(t)` is the close price of the S&P 500 on day :math:`t`.

==========================================
Inputs
==========================================

The model uses as input the previous 30-day returns of the S&P 500 as well as the past values
of the following volatility indicators:

* *RVOL*: The realized volatility of the S&P 500, calculated as the 30-day rolling sample standard deviation of the S&P 500 daily log returns.

* *VIX*: The `VIX index <https://www.cboe.com/us/indices/dashboard/vix/>`__ measures the 30-day implied volatility of S&P 500 options.

* *VVIX*: The `VVIX index <https://www.cboe.com/us/indices/dashboard/vvix/>`__ reflects the 30-day expected volatility of the VIX.

* *VXN*: The `VXN index <https://www.cboe.com/us/indices/dashboard/vxn/>`__ measures the 30-day implied volatility of NASDAQ 100 options.

* *GVZ*: The `GVZ index <https://www.cboe.com/us/indices/dashboard/gvz/>`__ measures the 30-day implied volatility of GLD options.

* *OVX*: The `OVX index <https://www.cboe.com/us/indices/dashboard/ovx/>`__ measures the 30-day implied volatility of USO options.

*RVOL* is a backward-looking indicator, as it measures the volatility over the past 30 days,
while *VIX*, *VVIX*, *VXN*, *GVZ*, and *OVX* are forward-looking indicators, as they reflect the market's
expectation of what the volatility will be over the next 30 days.

.. note::

    Note that we use the same inputs as in `[2] <file:///Users/flaviagiammarino/website/docs/blog/product/posts/cfc-tsf-equity-forecasting.html#references>`__,
    with the exception of the *PUTCALL* index, which we had to exclude as its historical time series is not publicly available.

******************************************
Code
******************************************

==========================================
Environment Set-Up
==========================================

We start by importing all the dependencies and setting up the SageMaker environment.

.. warning::

   To be able to run the code below, you need to have an active subscription to the
   CfC SageMaker algorithm. You can subscribe to a free trial from the
   `AWS Marketplace <https://aws.amazon.com/marketplace/pp/prodview-7s4giphluwgta>`__
   in order to get your Amazon Resource Name (ARN).
   In this post we use version 1.6 of the CfC SageMaker algorithm, which runs in the
   PyTorch 2.1.0 Python 3.10 deep learning container.

.. code:: python

    import io
    import sagemaker
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import yfinance as yf
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error, accuracy_score, f1_score

    # SageMaker session
    sagemaker_session = sagemaker.Session()

    # SageMaker role
    role = sagemaker.get_execution_role()

    # S3 bucket
    bucket = sagemaker_session.default_bucket()

    # EC2 instance
    instance_type = "ml.m5.4xlarge"

After that we define the neural network's context length and prediction length.
The context length is the number of past time steps used as input,
while the prediction length is the number of future time steps to be predicted.
We set both of them equal to 30 days, that is we use the previous 30 values
of the inputs to predict the next 30 values of the output.

.. code:: python

    # number of time steps used as input
    context_length = 30

    # number of time steps to output
    prediction_length = 30

==========================================
Data Preparation
==========================================
.. raw:: html

    Next, we download the daily close price time series from the 30<sup>th</sup> of June 2022 to
    the 29<sup>th</sup> of June 2024 with the
    <a href="https://github.com/ranaroussi/yfinance" target="_blank">Yahoo! Finance Python API</a>.

.. code:: python

    # download the data
    tickers = ["^SPX", "^VIX", "^VVIX", "^VXN", "^GVZ", "^OVX"]
    dataset = yf.download(" ".join(tickers), start="2022-06-30", end="2024-06-29")

    # extract the close prices
    dataset = dataset.loc[:, dataset.columns.get_level_values(0) == "Close"]
    dataset.columns = dataset.columns.get_level_values(1)

    # forward fill any missing values
    dataset.ffill(inplace=True)

We then calculate the S&P 500 30-day returns and 30-day realized volatility.

.. code:: python

    # calculate the returns
    dataset["Return30"] = np.log(dataset["^SPX"]).diff(periods=30)

    # calculate the realized volatility
    dataset["RVOL"] = np.log(dataset["^SPX"]).diff(periods=1).rolling(window=30).std(ddof=1)

    # drop the prices
    dataset.drop(labels=["^SPX"], axis=1, inplace=True)

    # drop the missing values
    dataset.dropna(inplace=True)

    # move the returns to the first column
    dataset = dataset[["Return30"] + dataset.columns.drop("Return30").tolist()]

The dataset contains 502 daily observations which, after dropping the missing values
resulting from the calculation of the returns of the realized volatility, are reduced to 472.

.. raw:: html

    <img
        id="cfc-tsf-forecasting-time-series"
        class="blog-post-image"
        alt="30-day returns, 30-day realized volatility and volatility indices from 2022-08-12 to 2024-06-29"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-forecasting/time_series_light.png
    />

    <p class="blog-post-image-caption">30-day returns, 30-day realized volatility and volatility indices from 2022-08-12 to 2024-06-29.</p>

We now proceed to renaming the columns in the format required by the CfC SageMaker algorithm,
where the output names should start with :code:`"y"` while the input names should start with :code:`"x"`.

.. code:: python

    dataset.columns = ["y"] + [f"x{i}" for i in range(dataset.shape[1] - 1)]

.. note::

    Note that the algorithm always uses the past values of the outputs as inputs,
    and there is therefore no need to include the outputs among the inputs when preparing the data for the model.

After that we split the data into a training set and a test set. The training set includes the first 70% of
the data (270 observations), while the test set includes the last 30% of the data (202 observations).

.. code:: python

    # define the size of the test set
    test_size = int(0.3 * len(dataset))

    # extract the training data
    training_dataset = dataset.iloc[:- test_size - context_length - prediction_length - 1]

    # extract the test data
    test_dataset = dataset.iloc[- test_size - context_length - prediction_length - 1:]

.. note::

    Note that the data is scaled internally by the algorithm, there is no need to scale the data beforehand.

==========================================
Training
==========================================
We now save the training data in S3, build the SageMaker estimator and run the training job.

.. code:: python

    # upload the training data to S3
    training_data = sagemaker_session.upload_string_as_file_body(
        body=training_dataset.to_csv(index=False),
        bucket=bucket,
        key="training_dataset.csv"
    )

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
            "hidden-size": 20,
            "backbone-layers": 1,
            "backbone-units": 40,
            "backbone-activation": "lecun",
            "backbone-dropout": 0,
            "minimal": True,
            "no-gate": True,
            "use-mixed": False,
            "use-ltc": False,
            "batch-size": 32,
            "lr": 0.0001,
            "lr-decay": 0.9999,
            "epochs": 800,
        }
    )

    # run the training job
    estimator.fit({"training": training_data})

.. note::

    Note that we are training a relatively small model with less than 5k parameters.

==========================================
Inference
==========================================
After the training job has been completed, we deploy the model to real-time endpoint that we can use for inference.

.. code:: python

    # define the endpoint inputs serializer
    serializer = sagemaker.serializers.CSVSerializer(content_type="text/csv")

    # define the endpoint outputs deserializer
    deserializer = sagemaker.base_deserializers.PandasDeserializer(accept="text/csv")

    # create the endpoint
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
    )

Once the endpoint has been created, we can generate the test set predictions.
As we used rolling (or overlapping) returns, we are only interested in the last
element of each predicted sequence (recall that we set the prediction length to 30 days,
the same as the horizon of the returns).

.. code:: python

    # create a list for storing the predictions
    predictions = []

    # loop across the dates
    for t in range(context_length, len(test_dataset) - prediction_length + 1):

        # extract the data up to day t - 1
        payload = test_dataset.iloc[t - context_length: t]

        # predict all rolling 30-day returns from day t to day t + 30
        response = sagemaker_session.sagemaker_runtime_client.invoke_endpoint(
            EndpointName=predictor.endpoint_name,
            ContentType="text/csv",
            Body=payload.to_csv(index=False)
        )
        response = deserializer.deserialize(response["Body"], content_type="text/csv")

        # extract the predicted 30-day return from day t to day t + 30
        prediction = response.iloc[-1:]

        # extract the date corresponding to day t + 30
        prediction.index = [test_dataset.index[t + prediction_length - 1]]

        # save the prediction
        predictions.append(prediction)

    # cast the predictions to data frame
    predictions = pd.concat(predictions)

    # add the actual values
    predictions["y"] = test_dataset["y"]

.. raw:: html

    <img
        id="cfc-tsf-forecasting-predictions"
        class="blog-post-image"
        alt="Actual and predicted 30-day returns from 2023-12-04 to 2024-06-28"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-forecasting/predictions_light.png
    />

    <p class="blog-post-image-caption">Actual and predicted 30-day returns from 2023-12-04 to 2024-06-28.</p>

We evaluate the test set predictions using the following metrics:

* *RMSE*: The root mean squared error of the predicted values of the returns.

* *MAE*: The mean absolute error of the predicted values of the returns.

* *Accuracy*: The accuracy of the predicted signs of the returns.

* *F1*: The F1 score of the predicted signs of the returns.

.. raw:: html

    <img
        id="cfc-tsf-forecasting-metrics"
        class="blog-post-image"
        alt="Performance metrics of predicted 30-day returns from 2023-12-04 to 2024-06-28"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-forecasting/metrics_light.png
    />

    <p class="blog-post-image-caption">Performance metrics of predicted 30-day returns from 2023-12-04 to 2024-06-28.</p>

We find that the model achieves a mean absolute error of 1.4% and
a mean directional accuracy of 97.5%.

.. raw:: html

    We now generate the out-of-sample forecasts, that is we predict the 30-day returns
    over 30 days beyond the end of the data (from the 29<sup>th</sup> of June 2024 to
    the 28<sup>th</sup> of July 2024).

.. note::

    In a real-life setting, we would retrain the model on all the available data
    (i.e. until the 28<sup>th</sup> of June 2024) before generating the out-of-sample
    forecasts. To avoid running a new training job, we simply use the existing endpoint,
    which uses the model trained on the data until the 8<sup>th</sup> of September 2023.

.. raw:: html

    <img
        id="cfc-tsf-forecasting-forecasts"
        class="blog-post-image"
        alt="30-day returns forecasts from 2024-06-29 to 2024-07-28"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-forecasting/forecasts_light.png
    />

    <p class="blog-post-image-caption">30-day returns forecasts from 2024-06-29 to 2024-07-28.</p>

******************************************
References
******************************************

[1] Kumbure, M.M., Lohrmann, C., Luukka, P. and Porras, J., (2022).
Machine learning techniques and data for stock market forecasting: A literature review.
*Expert Systems with Applications*, 197, p. 116659.
`doi: 10.1016/j.eswa.2022.116659 <https://doi.org/10.1016/j.eswa.2022.116659>`__.

[2] Campisi, G., Muzzioli, S. and De Baets, B., (2024).
A comparison of machine learning methods for predicting the direction of the US
stock market on the basis of volatility indices. *International Journal of Forecasting*, 40(3), pp. 869-880.
`doi: 10.1016/j.ijforecast.2023.07.002 <https://doi.org/10.1016/j.ijforecast.2023.07.002>`__.

[3] Funahashi, K.I. and Nakamura, Y., (1993). Approximation of dynamical systems by continuous
time recurrent neural networks. *Neural networks*, 6(6), pp.801-806.
`doi: 10.1016/S0893-6080(05)80125-X <https://doi.org/10.1016/S0893-6080(05)80125-X>`__.

[4] Hasani, R., Lechner, M., Amini, A., Liebenwein, L., Ray, A., Tschaikowski, M., Teschl, G. and Rus, D., (2022).
Closed-form continuous-time neural networks. *Nature Machine Intelligence*, 4(11), pp. 992-1003.
`doi: 10.1038/s42256-022-00556-7 <https://doi.org/10.1038/s42256-022-00556-7>`__.

[5] Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021).
Liquid time-constant networks. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(9), pp. 7657-7666.
`doi: 10.1609/aaai.v35i9.16936 <https://doi.org/10.1609/aaai.v35i9.16936>`__.
