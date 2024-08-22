.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Predicting stock market trends with SageMaker Autopilot
   :keywords: Amazon SageMaker, Time Series, AutoML, Forecasting, Stock Market

######################################################################################
Predicting stock market trends with SageMaker Autopilot
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
    We will use SageMaker AutoML V2 for building an ensemble of gradient boosting classifiers
    (XGBoost, LightGBM and CatBoost) to predict the direction of the S&P 500 (up or down)
    one day ahead using as input a set of technical indicators.
    </p>

    <p>
    We will download the S&P 500 daily time series from the
    2<sup>nd</sup> of August 2021 to the 31<sup>st</sup> of July 2024 from
    <a href="https://finance.yahoo.com" target="_blank">Yahoo! Finance</a>.
    We will train the model on the data up to the 2<sup>nd</sup> of May 2024 (training set),
    and validate the model on the subsequent 30 days of data up to the
    14<sup>th</sup> of June 2024 (validation set).
    We will then test the identified best model, i.e. the one with the best performance
    on the validation set, on the remaining 30 days of data up to the
    30<sup>th</sup> of July 2024 (test set).
    We will find that the model achieves an accuracy score of 63% and a
    ROC-AUC score of 80% over the test set.
    </p>

******************************************
Data
******************************************

==========================================
Outputs
==========================================
The model output (target) is the sign of the next day's price move of the S&P 500,
which is derived as follows

.. math::

    \begin{equation}
      Trend(t) =
        \begin{cases}
          1 & \text{if } P(t + 1) > P(t) \\
          0 & \text{if } P(t + 1) \le P(t)
        \end{cases}
    \end{equation}

where :math:`P(t)` is the close price of the S&P 500 on day :math:`t`.

==========================================
Inputs
==========================================
The model inputs (features) are the following technical indicators:

* Simple Moving Average

* Weighted Moving Average

* Momentum

* Stochastic K%

* Stochastic D%

* Relative Strength Index (RSI)

* Moving Average Convergence Divergence (MACD)

* Larry William’s R%

* Accumulation / Distribution (A/D) Oscillator

* Commodity Channel Index (CCI)

.. raw:: html

    <p>
    Note that we use the same technical indicators as in <a href="#references">[1]</a>.
    For the MACD we use periods of 12 days and 26 days, while for all the other indicators we use a period of 10 days.
    </p>

******************************************
Code
******************************************

==========================================
Environment Set-Up
==========================================

We start by importing all the dependencies and setting up the SageMaker environment.

.. note::

    We use the :code:`yfinance` library for downloading the S&P 500 daily time series and
    the :code:`pyti` library for calculating the technical indicators.

.. code:: python

    import warnings
    import io
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

==========================================
Data Preparation
==========================================

.. raw:: html

    <p>
    Next, we download the S&P 500 time series from the 2<sup>nd</sup> of August 2021 to the 31<sup>st</sup> of July 2024.
    The dataset contains 754 daily observations.
    </p>

.. code:: python

    # download the data
    dataset = yf.download(tickers="^SPX", start="2021-08-01", end="2024-08-01")

We then calculate the technical indicators.

.. code:: python

    # simple moving average
    dataset["Simple MA"] = simple_moving_average(
        data=dataset["Close"],
        period=10
    )

    # weighted moving average
    dataset["Weighted MA"] = weighted_moving_average(
        data=dataset["Close"],
        period=10
    )

    # momentum
    dataset["Momentum"] = momentum(
        data=dataset["Close"],
        period=10
    )

    # stochastic K%
    dataset["Stochastic K%"] = percent_k(
        data=dataset["Close"],
        period=10
    )

    # stochastic D%
    dataset["Stochastic D%"] = percent_d(
        data=dataset["Close"],
        period=10
    )

    # relative strength index
    dataset["RSI"] = relative_strength_index(
        data=dataset["Close"],
        period=10
    )

    # moving average convergence divergence
    dataset["MACD"] = moving_average_convergence_divergence(
        data=dataset["Close"],
        short_period=12,
        long_period=26
    )

    # Larry William’s R%
    dataset["LW R%"] = williams_percent_r(
        close_data=dataset["Close"],
    )

    # accumulation / distribution oscillator
    dataset["A/D Oscillator"] = accumulation_distribution(
        close_data=dataset["Close"],
        low_data=dataset["Low"],
        high_data=dataset["High"],
        volume=dataset["Volume"]
    )

    # commodity channel index
    dataset["CCI"] = commodity_channel_index(
        close_data=dataset["Close"],
        low_data=dataset["Low"],
        high_data=dataset["High"],
        period=10
    )

We also calculate the binary labels, which are equal to 1 when the price of the S&P 500 goes up on the next day,
and equal to 0 otherwise.

.. code:: python

    # derive the class labels (up = 1, down = 0)
    dataset.insert(0, "Trend", (dataset["Close"].shift(periods=-1) > dataset["Close"]).mask(dataset["Close"].shift(periods=-1).isna()).astype(float))

After dropping the missing values resulting from the calculation of the technical indicators
and of the binary labels, the number of daily observations is reduced to 728.

.. code:: python

    # drop the unnecessary columns
    dataset.drop(labels=["Close", "Open", "High", "Low", "Volume", "Adj Close"], axis=1, inplace=True)

    # drop the missing values
    dataset.dropna(inplace=True)

.. code:: python

    dataset.shape

.. code-block:: console

    (728, 11)

.. code:: python

    dataset.head()

.. raw:: html

    <img
        id="equity-trend-prediction-automl-dataset-head"
        class="blog-post-image"
        alt="First 3 rows of dataset"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-trend-prediction-automl/dataset_head_light.png
        style="width:100%"
    />

.. code:: python

    dataset.tail()

.. raw:: html

    <img
        id="equity-trend-prediction-automl-dataset-tail"
        class="blog-post-image"
        alt="Last 3 rows of dataset"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-trend-prediction-automl/dataset_tail_light.png
        style="width:100%"
    />

.. raw:: html

    <img
        id="equity-trend-prediction-automl-time-series"
        class="blog-post-image"
        alt="S&P 500 with technical indicators from 2021-09-07 to 2024-07-30"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-trend-prediction-automl/time_series_light.png
    />

    <p class="blog-post-image-caption">S&P 500 with technical indicators from 2021-09-07 to 2024-07-30.</p>

We use the last 30 days of data for testing, the prior 30 days of data for validation,
and all the previous data for training.

.. code:: python

    # define the size of the test set
    test_size = 30

    # extract the training data
    training_dataset = dataset.iloc[:- 2 * test_size]

    # extract the validation data
    validation_dataset = dataset.iloc[- 2 * test_size: - test_size]

    # extract the test data
    test_dataset = dataset.iloc[- test_size:]

We save the training, validation and test data to S3 in CSV format.

.. code:: python

    # save the training data to S3
    training_data = session.upload_string_as_file_body(
        body=training_dataset.to_csv(index=False),
        bucket=bucket,
        key="data/train.csv"
    )

    # save the validation data to S3
    validation_data = session.upload_string_as_file_body(
        body=validation_dataset.to_csv(index=False),
        bucket=bucket,
        key="data/valid.csv"
    )

    # save the test data to S3
    test_data = session.upload_string_as_file_body(
        body=test_dataset.drop(labels=["Trend"], axis=1).to_csv(index=False, header=False),
        bucket=bucket,
        key="data/test.csv"
    )

==========================================
Model Selection
==========================================

We now run an AutoML V2 job to find the best ensemble of gradient boosting classifiers (XGBoost, LightGBM and CatBoost).
In the interest of time, we limit the number of candidate models to 10.

.. code:: python

    # define the AutoML job configuration
    automl = sagemaker.automl.automlv2.AutoMLV2(
        problem_config=sagemaker.automl.automlv2.AutoMLTabularConfig(
            target_attribute_name="Trend",
            algorithms_config=["xgboost", "lightgbm", "catboost"],
            mode="ENSEMBLING",
            problem_type="BinaryClassification",
            max_candidates=10,
        ),
        output_path=f"s3://{bucket}/output/",
        job_objective={"MetricName": "Accuracy"},
        base_job_name="equity-trend-automl",
        role=role,
        sagemaker_session=session,
    )

    # run the AutoML job
    automl.fit(
        inputs=[
            sagemaker.automl.automlv2.AutoMLDataChannel(
                s3_data_type="S3Prefix",
                s3_uri=training_data,
                channel_type="training",
                compression_type=None,
                content_type="text/csv;header=present"
            ),
            sagemaker.automl.automlv2.AutoMLDataChannel(
                s3_data_type="S3Prefix",
                s3_uri=validation_data,
                channel_type="validation",
                compression_type=None,
                content_type="text/csv;header=present"
            ),
        ]
    )

The AutoML V2 job generates numerous outputs, including an insight report for each
model in the ensemble, and an explainability report with the feature importance
for the overall ensemble, which are saved in S3.

.. raw:: html

    <img
        id="equity-trend-prediction-automl-importances"
        class="blog-post-image"
        alt="Feature importance."
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-trend-prediction-automl/importances_light.png
    />

    <p class="blog-post-image-caption">Feature importance.</p>

==========================================
Model Evaluation
==========================================

After the AutoML V2 job has been completed, we run a batch transform job on
the test data using the best model. Note that we configure the model such that
it outputs the predicted probabilities in addition to the predicted labels.

.. code:: python

    # create the model
    model = automl.create_model(
        name="equity-trend-model",
        sagemaker_session=session,
        inference_response_keys=["probabilities", "labels", "predicted_label", "probability"]
    )

    # create the transformer
    transformer = model.transformer(
        instance_count=1,
        instance_type="ml.m5.2xlarge",
    )

    # run the transform job
    transformer.transform(
        data=test_data,
        content_type="text/csv",
    )

The results of the batch transform job are saved to a CSV file in S3 with the same name as the
input CSV file but with the :code:`".out"` file extension.

.. code:: python

    # download the predictions from S3
    predictions = session.read_s3_file(
        bucket=bucket,
        key_prefix=f"{transformer.latest_transform_job.name}/test.csv.out"
    )

    # cast the predictions to data frame
    predictions = pd.read_csv(io.StringIO(predictions), header=None)

.. code:: python

    predictions.shape

.. code-block:: console

    (30, 4)

.. code:: python

    predictions.head()

.. raw:: html

    <img
        id="equity-trend-prediction-automl-raw-predictions-head"
        class="blog-post-image"
        alt="First 3 rows of raw predictions"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-trend-prediction-automl/raw_predictions_head_light.png
        style="width:100%"
    />

.. code:: python

    predictions.tail()

.. raw:: html

    <img
        id="equity-trend-prediction-automl-raw-predictions-tail"
        class="blog-post-image"
        alt="Last 3 rows of raw predictions"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-trend-prediction-automl/raw_predictions_tail_light.png
        style="width:100%"
    />

For convenience, we include the ground truth class labels in the same data frame as the predicted class labels and class probabilities.

.. code:: python

    # extract the predicted probabilities
    predictions["Class 0 Probability"] = predictions["probabilities"].apply(lambda x: json.loads(x)[1])
    predictions["Class 1 Probability"] = predictions["probabilities"].apply(lambda x: json.loads(x)[0])
    predictions["Predicted Trend"] = predictions[["Class 0 Probability", "Class 1 Probability"]].apply(lambda x: np.argmax(x), axis=1)

    # add the dates
    predictions.index = test_dataset.index

    # add the ground truth labels
    predictions["Actual Trend"] = test_dataset["Trend"].astype(int)

    # drop the unnecessary columns
    predictions = predictions[["Class 0 Probability", "Class 1 Probability", "Predicted Trend", "Actual Trend"]]

.. code:: python

    predictions.shape

.. code-block:: console

    (30, 6)

.. code:: python

    predictions.head()

.. raw:: html

    <img
        id="equity-trend-prediction-automl-predictions-head"
        class="blog-post-image"
        alt="First 3 rows of predictions"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-trend-prediction-automl/predictions_head_light.png
        style="width:100%"
    />

.. code:: python

    predictions.tail()

.. raw:: html

    <img
        id="equity-trend-prediction-automl-predictions-tail"
        class="blog-post-image"
        alt="Last 3 rows of predictions"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-trend-prediction-automl/predictions_tail_light.png
        style="width:100%"
    />

We can finally calculate the classification metrics of the test set predictions.

.. code:: python

    # calculate the classification metrics
    metrics = pd.DataFrame(
        data={
            "Accuracy": accuracy_score(y_true=predictions["Actual Trend"], y_pred=predictions["Predicted Trend"]),
            "ROC-AUC": roc_auc_score(y_true=predictions["Actual Trend"], y_score=predictions["Class 1 Probability"]),
            "Precision": precision_score(y_true=predictions["Actual Trend"], y_pred=predictions["Predicted Trend"]),
            "Recall": recall_score(y_true=predictions["Actual Trend"], y_pred=predictions["Predicted Trend"]),
            "F1": f1_score(y_true=predictions["Actual Trend"], y_pred=predictions["Predicted Trend"]),
        },
        index=["Value"]
    ).transpose().reset_index().rename(columns={"index": "Metric"})

.. raw:: html

    <img
        id="equity-trend-prediction-automl-metrics"
        class="blog-post-image"
        alt="Performance metrics of predicted S&P 500 directional moves over the test set (from 2024-06-17 to 2024-07-30)"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-trend-prediction-automl/metrics_light.png
        style="width:85%"
    />

    <p class="blog-post-image-caption">Performance metrics of predicted S&P 500 directional moves over the test set (from 2024-06-17 to 2024-07-30).</p>

We find that the model achieves an accuracy score of 63% and a ROC-AUC score of 80% on the test set.

.. raw:: html

    <img
        id="equity-trend-prediction-automl-roc-auc"
        class="blog-post-image"
        alt="ROC curve of predicted S&P 500 directional moves over the test set (from 2024-06-17 to 2024-07-30)"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-trend-prediction-automl/roc_auc_light.png
    />

    <p class="blog-post-image-caption">ROC curve of predicted S&P 500 directional moves over the test set (from 2024-06-17 to 2024-07-30).</p>

We can additionally calculate the confusion matrix of the test set predictions.

.. code:: python

    # calculate the confusion matrix
    matrix = pd.crosstab(
        index=predictions["Actual Trend"],
        columns=predictions["Predicted Trend"],
    )

This shows that the model tends to underestimate the number of up moves over the considered time period.

.. raw:: html

    <img
        id="equity-trend-prediction-automl-matrix"
        class="blog-post-image"
        alt="Confusion matrix of predicted S&P 500 directional moves over the test set (from 2024-06-17 to 2024-07-30)"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-trend-prediction-automl/matrix_light.png
        style="width:85%"
    />

    <p class="blog-post-image-caption">Confusion matrix of predicted S&P 500 directional moves over the test set (from 2024-06-17 to 2024-07-30).</p>

After the analysis has been completed, we can delete the model.

.. code:: python

    # delete the model
    transformer.delete_model()

.. tip::

    A Python notebook with the full code is available in our
    `GitHub <https://github.com/fg-research/blog/blob/master/equity-trend-prediction-automl>`__
    repository.

******************************************
References
******************************************

[1] Kara, Y., Boyacioglu, M. A., & Baykan, Ö. K. (2011).
Predicting direction of stock price index movement using artificial neural networks and support vector machines:
The sample of the Istanbul Stock Exchange. *Expert Systems with Applications*, 38(5), 5311-5319.
`doi: doi:10.1016/j.eswa.2010.10.027 <https://doi.org/doi:10.1016/j.eswa.2010.10.027>`__.
