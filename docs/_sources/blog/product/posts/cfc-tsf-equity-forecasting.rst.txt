.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Forecasting Stock Returns with Liquid Neural Networks
   :keywords: Amazon SageMaker, Time Series, Liquid Neural Networks, Forecasting, Stock Market

###########################################################################################
Forecasting stock returns with liquid neural networks using the CfC SageMaker Algorithm
###########################################################################################

.. raw:: html

    <p>
    Stock return forecasting has been extensively studied by both academic researchers and
    industry practitioners. Numerous machine learning models have been proposed for this purpose,
    ranging from simple linear regressions to complex deep learning models <a href="#references">[1]</a>.
    In this post, we examine the performance of liquid neural networks (LNNs) <a href="#references">[5]</a>
    <a href="#references">[6]</a>, a new neural network architecture for sequential data.
    </p>

    <p>
    LNNs belong to the class of continuous-time recurrent neural networks (CT-RNNs)
    <a href="#references">[3]</a> <a href="#references">[4]</a>, where the evolution of the
    hidden state over time follows an Ordinary Differential Equation (ODE).
    LNNs are based on the Liquid Time Constant (LTC) ODE <a href="#references">[5]</a>,
    where both the derivative and the time constant of the hidden state are determined
    by a neural network. As a result of their higher flexibility and expressiveness,
    LNNs can capture more complex patterns and relationships within
    the data than other RNNs and, as a result, often
    outperform modern deep learning models on time-series prediction tasks.
    </p>

    <p>
    LNNs were initially implemented as LTC networks or LTCs <a href="#references">[5]</a>.
    Similar to other CT-RNNs, LTCs use a numerical solver for finding the ODE solution,
    resulting in slow training and inference performance.
    In this post, we focus on the closed-form continuous-depth (CfC) implementation of LNNs
    <a href="#references">[6]</a>. CfCs implement an approximate closed-form solution
    of the LTC ODE and, as a result, are significantly faster than LTCs and other CT-RNNs.
    </p>

    <p>
    We will use our Amazon SageMaker implementation of CfCs for probabilistic time series
    forecasting, the <a href="https://fg-research.com/algorithms/time-series-forecasting/index.html#cfc-sagemaker-algorithm" target="_blank"> CfC SageMaker algorithm</a>.
    We will forecast the conditional mean and the conditional standard deviation of the 30-day returns of
    the S&P 500 using as input the S&P 500 realized volatility as well as different implied volatility indices,
    as in <a href="#references">[2]</a>.
    </p>

    <p>
    We will use the daily close prices from the 30<sup>th</sup> of June 2022 to
    the 28<sup>th</sup> of June 2024, which we will download from
    <a href="https://finance.yahoo.com" target="_blank">Yahoo! Finance</a>.
    We will train the model on the data up to the 8<sup>th</sup> of September 2023,
    and use the trained model to predict the subsequent data up to the 28<sup>th</sup> of June 2024.
    We will find that the CfC SageMaker algorithm achieves a mean absolute error of 1.4% and
    a mean directional accuracy of 95.8%.
    </p>

******************************************
Data
******************************************

==========================================
Outputs
==========================================

The model outputs are the 30-day returns of the S&P 500, which are calculated as follows

.. math::

    y(t) = \ln{P(t)} - \ln{P(t - 30)}

for each day :math:`t`, where :math:`P(t)` is the close price of the S&P 500 on day :math:`t`.
We will use a prediction length of 30 days, meaning that the model will output the 30-day returns
over the subsequent 30 days. Given that we use overlapping (or rolling) returns, the predicted
30-day return from day :math:`t` to day :math:`t + 30` is the last return in the output sequence.

==========================================
Inputs
==========================================

The model uses as input the previous 30-day returns of the S&P 500, as well as the past values
of the following volatility indicators:

* The realized volatility of the S&P 500 (*RVOL*), which is calculated as the 30-day rolling sample standard deviation of the S&P 500 daily returns.

* The *VIX* index, which measures the 30-day implied volatility of S&P 500 options.

* The *VVIX* index, which reflects the 30-day expected volatility of the VIX.

* The *VXN* index, which measures the 30-day implied volatility of NASDAQ 100 options.

* The *GVZ* index, which measures the 30-day implied volatility of SPDR Gold Shares ETF (GLD) options.

* The *OVX* index, which measures the 30-day implied volatility of United States Oil Fund (USO) options.

*RVOL* is a backward-looking indicator, as it estimates the volatility over the past 30 days,
while *VIX*, *VVIX*, *VXN*, *GVZ*, and *OVX* are forward-looking indicators, as they reflect the market's
expectation of what the volatility will be over the next 30 days.

.. raw:: html

    <p>
    Note that we use the same inputs as in <a href="#references">[2]</a>, with the exception of the
    <i>PUTCALL</i> index, which we had to exclude as its historical time series is not publicly available.
    Note also that, as discussed in <a href="#references">[2]</a>, we exclude the (short-term) term
    structure of the VIX index (VIX9D, VIX3M, VIX6M) as the different tenor points are highly correlated
    with each other and with the VIX index, resulting in high multi-collinearity and low predictive power.
    </p>

We will use a context length of 30 days, meaning that the model will use as input the 30-day returns
and the volatility indicators over the previous 30 days in order to predict the 30-day returns over
the subsequent 30 days.

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

After that we define the neural network's *context length* and *prediction length*.
The context length is the number of past time steps used as input,
while the prediction length is the number of future time steps to be predicted.
We set both of them equal to 30 days, that is we use the previous 30 values
of the inputs and output to predict the subsequent 30 values of the output.

.. code:: python

    # number of time steps used as input
    context_length = 30

    # number of time steps to output
    prediction_length = 30

We also define all the remaining hyperparameters of the CfC network.
Note that we use a relatively small model with less than 5k parameters.
A detailed description of the model architecture and of its hyperparameters
is provided in our `GitHub repository <https://github.com/fg-research/cfc-tsf-sagemaker>`__.

.. code:: python

    # neural network hyperparameters
    hyperparameters = {
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


==========================================
Data Preparation
==========================================

.. raw:: html

    <p>
    Next, we download the daily close price time series from the 30<sup>th</sup> of June 2022 to
    the 28<sup>th</sup> of June 2024 from <a href="https://finance.yahoo.com" target="_blank">Yahoo! Finance</a>
    using the <a href="https://github.com/ranaroussi/yfinance" target="_blank">Yahoo! Finance Python API</a>.
    The dataset contains 502 daily observations.
    </p>

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

After dropping the missing values resulting from the calculation of the returns and of the realized volatility,
the number of daily observations is reduced to 472.

.. raw:: html

    <img
        id="cfc-tsf-forecasting-time-series"
        class="blog-post-image"
        alt="30-day returns, 30-day realized volatility and volatility indices from 2022-08-12 to 2024-06-28"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-forecasting/time_series_light.png
    />

    <p class="blog-post-image-caption">30-day returns, 30-day realized volatility and volatility indices from 2022-08-12 to 2024-06-28.</p>

We now proceed to renaming the columns in the format expected by the CfC SageMaker algorithm,
where the output names should start with :code:`"y"` and the input names should start with :code:`"x"`.

.. code:: python

    dataset.columns = ["y"] + [f"x{i}" for i in range(dataset.shape[1] - 1)]

.. note::

    Note that the algorithm's code always includes the past values of the outputs
    among the inputs and, therefore, there is no need to add the lagged values of
    the outputs when preparing the data for the model.

==========================================
Testing
==========================================

For the purpose of validating the model, we split the data into a training set and a test set.
The training set includes the first 70% of the data, while the test set
includes the last 30% of the data.

.. code:: python

    # define the size of the test set
    test_size = int(0.3 * len(dataset))

    # extract the training data
    training_dataset = dataset.iloc[:- test_size - context_length - prediction_length - 1]

    # extract the test data
    test_dataset = dataset.iloc[- test_size - context_length - prediction_length - 1:]

.. note::

    Note that the data is scaled internally by the algorithm, there is no need to scale the data beforehand.

We now save the training data in S3, build the SageMaker estimator and run the training job.

.. code:: python

    # upload the training data to S3
    training_data = sagemaker_session.upload_string_as_file_body(
        body=training_dataset.to_csv(index=False),
        bucket=bucket,
        key="training_data.csv"
    )

    # create the estimator
    estimator = sagemaker.algorithm.AlgorithmEstimator(
        algorithm_arn=algo_arn,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        input_mode="File",
        sagemaker_session=sagemaker_session,
        hyperparameters=hyperparameters
    )

    # run the training job
    estimator.fit({"training": training_data})

After the training job has been completed, we deploy the model to a real-time endpoint that we can use for inference.

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
Given that the returns are overlapping, we are only interested in the last
element of each predicted sequence (recall that the prediction length is 30 days,
the same as the horizon of the returns).

.. code:: python

    # create a list for storing the predictions
    predictions = []

    # loop across the dates
    for t in range(context_length, len(test_dataset) - prediction_length + 1):

        # extract the inputs
        payload = test_dataset.iloc[t - context_length: t]

        # invoke the endpoint
        response = sagemaker_session.sagemaker_runtime_client.invoke_endpoint(
            EndpointName=predictor.endpoint_name,
            ContentType="text/csv",
            Body=payload.to_csv(index=False)
        )

        # deserialize the endpoint response
        response = deserializer.deserialize(response["Body"], content_type="text/csv")

        # extract the predicted 30-day return
        prediction = response.iloc[-1:]

        # extract the date corresponding to the predicted 30-day return
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

    <p class="blog-post-image-caption">Actual and predicted 30-day returns over the test set (from 2023-12-04 to 2024-06-28).</p>

We evaluate the test set predictions using the following metrics:

* The root mean squared error (*RMSE*) of the predicted values of the returns.

* The mean absolute error (*MAE*) of the predicted values of the returns.

* The *accuracy* of the predicted signs of the returns.

* The *F1* score of the predicted signs of the returns.

.. code:: python

    # calculate the model performance metrics
    metrics = pd.DataFrame(
        columns=["Metric", "Value"],
        data=[
            {"Metric": "RMSE", "Value": root_mean_squared_error(y_true=predictions["y"], y_pred=predictions["y_mean"])},
            {"Metric": "MAE", "Value": mean_absolute_error(y_true=predictions["y"], y_pred=predictions["y_mean"])},
            {"Metric": "Accuracy", "Value": accuracy_score(y_true=predictions["y"] > 0, y_pred=predictions["y_mean"] > 0)},
            {"Metric": "F1", "Value": f1_score(y_true=predictions["y"] > 0, y_pred=predictions["y_mean"] > 0)},
        ]
    )

We find that the model achieves a mean absolute error of 1.4% and a mean directional accuracy of 95.8% on the test set.

.. raw:: html

    <img
        id="cfc-tsf-forecasting-metrics"
        class="blog-post-image"
        alt="Performance metrics of predicted 30-day returns over the test set (from 2023-12-04 to 2024-06-28)"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-forecasting/metrics_light.png
    />

    <p class="blog-post-image-caption">Performance metrics of predicted 30-day returns over the test set (from 2023-12-04 to 2024-06-28).</p>

We can now delete the model and the endpoint.

.. code:: python

    # delete the model
    predictor.delete_model()

    # delete the endpoint
    predictor.delete_endpoint(delete_endpoint_config=True)

==========================================
Forecasting
==========================================

.. raw:: html

    <p>
    We now retrain the model using all the available data, and generate the out-of-sample forecasts,
    that is we predict the 30-day returns over 30 (business) days beyond the current date (2024-06-28).
    </p>

.. code:: python

    # upload the training data to S3
    data = sagemaker_session.upload_string_as_file_body(
        body=dataset.to_csv(index=False),
        bucket=bucket,
        key="dataset.csv"
    )

    # create the estimator
    estimator = sagemaker.algorithm.AlgorithmEstimator(
        algorithm_arn=algo_arn,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        input_mode="File",
        sagemaker_session=sagemaker_session,
        hyperparameters=hyperparameters
    )

    # run the training job
    estimator.fit({"training": data})

Given that we only need a single predicted 30-day sequence, we use batch transform for generating the forecasts.
The forecasts are saved to a CSV file in S3 with the same name as the input CSV file but with the `".out"` file extension.

.. code:: python

    # upload the input data to S3
    inputs = sagemaker_session.upload_string_as_file_body(
        body=dataset.iloc[- context_length:].to_csv(index=False),
        bucket=bucket,
        key="inputs.csv"
    )

    # create the transformer
    transformer = estimator.transformer(
        instance_count=1,
        instance_type=instance_type,
    )

    # run the transform job
    transformer.transform(
        data=inputs,
        content_type="text/csv",
    )

After the batch transform job has been completed, we can load the forecasts from S3.

.. code:: python

    # download the forecasts from S3
    forecasts = sagemaker_session.read_s3_file(
        bucket=bucket,
        key_prefix=f"{transformer.latest_transform_job.name}/inputs.csv.out"
    )

    # cast the forecasts to data frame
    forecasts = pd.read_csv(io.StringIO(forecasts), dtype=float).dropna()

    # add the forecast dates
    forecasts.index = pd.date_range(
        start=dataset.index[-1] + pd.Timedelta(days=1),
        periods=prediction_length,
        freq="B"
    )

.. raw:: html

    <img
        id="cfc-tsf-forecasting-forecasts"
        class="blog-post-image"
        alt="30-day returns out-of-sample forecasts (from 2024-07-01 to 2024-08-09)"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-forecasting/forecasts_light.png
    />

    <p class="blog-post-image-caption">30-day returns out-of-sample forecasts (from 2024-07-01 to 2024-08-09).</p>

We can now delete the model.

.. code:: python

    # delete the model
    transformer.delete_model()

.. tip::

    You can download the
    `notebook <https://github.com/fg-research/cfc-tsf-sagemaker/blob/master/examples/SPX.ipynb>`__
    with the full code from our
    `GitHub <https://github.com/fg-research/cfc-tsf-sagemaker>`__
    repository.

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

[3] Chow, T.W. and Li, X.D., 2000. Modeling of continuous time dynamical systems with input by
recurrent neural networks. *IEEE Transactions on Circuits and Systems I: Fundamental Theory and Applications*,
47(4), pp.575-578. `doi: 10.1109/81.841860 <https://doi.org/10.1109/81.841860>`__.

[4] Funahashi, K.I. and Nakamura, Y., (1993). Approximation of dynamical systems by continuous
time recurrent neural networks. *Neural networks*, 6(6), pp.801-806.
`doi: 10.1016/S0893-6080(05)80125-X <https://doi.org/10.1016/S0893-6080(05)80125-X>`__.

[5] Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021).
Liquid time-constant networks. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(9), pp. 7657-7666.
`doi: 10.1609/aaai.v35i9.16936 <https://doi.org/10.1609/aaai.v35i9.16936>`__.

[6] Hasani, R., Lechner, M., Amini, A., Liebenwein, L., Ray, A., Tschaikowski, M., Teschl, G. and Rus, D., (2022).
Closed-form continuous-time neural networks. *Nature Machine Intelligence*, 4(11), pp. 992-1003.
`doi: 10.1038/s42256-022-00556-7 <https://doi.org/10.1038/s42256-022-00556-7>`__.
