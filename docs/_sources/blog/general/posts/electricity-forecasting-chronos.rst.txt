.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Forecasting electricity prices with Amazon Chronos
   :keywords: Large Language Models, Transformers, Time Series, Forecasting, Energy

######################################################################################
Forecasting electricity prices with Amazon Chronos
######################################################################################

Chronos is a foundational model for one-shot probabilistic forecasting of univariate time series.
The model converts time series data into a sequence of tokens through scaling and quantization.
The scaling is performed by dividing each time series by its mean absolute value, while
the quantization process uses uniform binning for mapping the scaled time series values
to a discrete set of tokens.

The tokenized time series is then used as input into a large language model (LLM).
Chronos was pre-trained using the T5 model, but it supports any other LLM architecture.
Chronos is trained in a self-supervised manner by minimizing the cross-entropy loss between
the actual and predicted next token, as it is standard for LLMs.

At inference time, Chronos takes as input a sequence of tokens (a context window) and returns
a random sample from the predicted distribution of the next token. The generated tokens are
then converted back to time series values by inverting the quantization and scaling transformations.

In this post, we demonstrate how to use Chronos for one-step-ahead forecasting.
We will use the US average electricity price monthly time series from November 1978 to July 2024,
which we will download from the FRED database, and generate one-month-ahead forecasts from August 2014 to July 2024.
We will use expanding context windows, that is on each month we will provide Chronos
all the data up to the month, and generate the forecast for the next month.

We will compare Chronos' one-shot forecasts to the rolling forecasts of a SARIMA model which is
re-trained on each month using the same expanding windows that were provided to Chronos as context.
We will find that the Chronos model and the SARIMA model have comparable performance.

******************************************
Code
******************************************

We start by installing and importing all the dependencies.

.. code:: python

    pip install git+https://github.com/amazon-science/chronos-forecasting.git fredapi pmdarima

.. code:: python

    import warnings
    import transformers
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import datetime
    from chronos import ChronosPipeline
    from pmdarima.arima import auto_arima
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from tqdm import tqdm
    from fredapi import Fred
    from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, root_mean_squared_error

.. raw:: html

    <p>
    Next, we download the time series from the <a href=https://fred.stlouisfed.org/ target="_blank">FRED database</a>.
    We use the <a href="https://github.com/mortada/fredapi" target="_blank">Python API for FRED</a> for downloading the data.
    </p>

.. tip::
    If you don’t have a FRED API key, you can request one for free at `this link <http://api.stlouisfed.org/api_key.html>`__.

.. code:: python

    # set up the FRED API
    fred = Fred(api_key_file="api_key.txt")

    # define the time series ID
    series = "APU000072610"

    # download the time series
    data = fred.get_series(series).rename(series).ffill()

The time series includes 549 monthly observations from November 1978 to July 2024.
The time series had one missing value in September 1985, which we forward filled with the previous value.

.. raw:: html

    <img
        id="electricity-forecasting-chronos-time-series"
        class="blog-post-image"
        alt="US average electricity price from November 1978 to July 2024"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/electricity-forecasting-chronos/time_series_light.png
    />

    <p class="blog-post-image-caption">US average electricity price from November 1978 to July 2024.</p>

We generate the forecasts over a 10-year period (120 months) from August 2014 to July 2024.

.. code:: python

    # date of first forecast
    start_date = "2014-08-01"

    # date of last forecast
    end_date = "2024-07-01"

==========================================
SARIMA
==========================================
We use the :code:`pmdarima` library for finding the best order of the SARIMA model using the data up to July 2014.

.. code:: python

    # find the best order of the SARIMA model
    best_sarima_model = auto_arima(
        y=data[data.index < start_date],
        start_p=0,
        start_q=0,
        start_P=0,
        start_Q=0,
        m=12,
        seasonal=True,
    )

.. raw:: html

    <img
        id="electricity-forecasting-chronos-sarima-results"
        class="blog-post-image"
        alt="SARIMA estimation results."
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/electricity-forecasting-chronos/sarimax_results.png
    />

    <p class="blog-post-image-caption">SARIMA estimation results.</p>

For each month in the considered time window, we train the SARIMA model with the identified best order
on all the data up to that month, and generate the forecast for the next month.

.. code:: python

    # create a list for storing the forecasts
    sarima_forecasts = []

    # loop across the dates
    for t in tqdm(range(data.index.get_loc(start_date), data.index.get_loc(end_date) + 1)):

        # extract the training data
        context = data.iloc[:t]

        # train the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sarima_model = SARIMAX(
                endog=context,
                order=best_sarima_model.order,
                seasonal_order=best_sarima_model.seasonal_order,
                trend="c" if best_sarima_model.with_intercept else None,
            ).fit(disp=0)

        # generate the one-step-ahead forecast
        sarima_forecast = sarima_model.get_forecast(steps=1)

        # save the forecast
        sarima_forecasts.append({
            "date": data.index[t],
            "actual": data.values[t],
            "mean": sarima_forecast.predicted_mean.item(),
            "std": sarima_forecast.var_pred_mean.item() ** 0.5,
        })

    # cast the forecasts to data frame
    sarima_forecasts = pd.DataFrame(sarima_forecasts)

.. raw:: html

    <img
        id="electricity-forecasting-chronos-sarima-forecasts"
        class="blog-post-image"
        alt="SARIMA forecasts from August 2014 to July 202."
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/electricity-forecasting-chronos/sarima_forecasts_light.png
    />

    <p class="blog-post-image-caption">SARIMA forecasts from August 2014 to July 2024.</p>

We find that the SARIMA model achieves an RMSE of 0.001364 and a MAE of 0.001067.

.. code:: python

    # calculate the error metrics
    sarima_metrics = pd.DataFrame(
        columns=["Metric", "Value"],
        data=[
            {"Metric": "RMSE", "Value": root_mean_squared_error(y_true=sarima_forecasts["actual"], y_pred=sarima_forecasts["mean"])},
            {"Metric": "MAE", "Value": mean_absolute_error(y_true=sarima_forecasts["actual"], y_pred=sarima_forecasts["mean"])},
        ]
    ).set_index("Metric")


.. raw:: html

    <img
        id="electricity-forecasting-chronos-sarima-metrics"
        class="blog-post-image"
        alt="SARIMA forecast errors from August 2014 to July 202."
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/electricity-forecasting-chronos/sarima_metrics_light.png
    />

    <p class="blog-post-image-caption">SARIMA forecast errors from August 2014 to July 2024.</p>

==========================================
Chronos
==========================================
We use the t5-large version of Chronos, which includes approximately 710 million parameters.

.. code:: python

    # instantiate the model
    chronos_model = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-large",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )

For each month in the considered time window, we use as context window all the data up to that month,
and generate 100 samples from the predicted distribution for the next month.
We use the mean of the distribution as point forecast, as in the SARIMA model.

.. note::

    Note that, as Chronos is a generative model, different random seeds and different numbers of
    samples result in slightly different forecasts.

.. code:: python

    # create a list for storing the forecasts
    chronos_forecasts = []

    # loop across the dates
    for t in tqdm(range(data.index.get_loc(start_date), data.index.get_loc(end_date) + 1)):

        # extract the context window
        context = data.iloc[:t]

        # generate the one-step-ahead forecast
        transformers.set_seed(42)
        chronos_forecast = chronos_model.predict(
            context=torch.from_numpy(context.values),
            prediction_length=1,
            num_samples=100
        ).detach().cpu().numpy().flatten()

        # save the forecast
        chronos_forecasts.append({
            "date": data.index[t],
            "actual": data.values[t],
            "mean": np.mean(chronos_forecast),
            "std": np.std(chronos_forecast, ddof=1),
        })

    # cast the forecasts to data frame
    chronos_forecasts = pd.DataFrame(chronos_forecasts)


.. raw:: html

    <img
        id="electricity-forecasting-chronos-chronos-forecasts"
        class="blog-post-image"
        alt="Chronos forecasts from August 2014 to July 202."
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/electricity-forecasting-chronos/chronos_forecasts_light.png
    />

    <p class="blog-post-image-caption">Chronos forecasts from August 2014 to July 2024.</p>

.. code:: python

    # calculate the error metrics
    chronos_metrics = pd.DataFrame(
        columns=["Metric", "Value"],
        data=[
            {"Metric": "RMSE", "Value": root_mean_squared_error(y_true=chronos_forecasts["actual"], y_pred=chronos_forecasts["mean"])},
            {"Metric": "MAE", "Value": mean_absolute_error(y_true=chronos_forecasts["actual"], y_pred=chronos_forecasts["mean"])},
        ]
    ).set_index("Metric")


.. raw:: html

    <img
        id="electricity-forecasting-chronos-chronos-metrics"
        class="blog-post-image"
        alt="Chronos forecast errors from August 2014 to July 202."
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/electricity-forecasting-chronos/chronos_metrics_light.png
    />

    <p class="blog-post-image-caption">Chronos forecast errors from August 2014 to July 2024.</p>

We find that the Chronos model achieves an RMSE of 0.001443 and a MAE of 0.001105.

.. tip::

    A Python notebook with the full code is available in our `GitHub <https://github.com/fg-research/blog/blob/master/electricity-forecasting-chronos>`__
    repository.

******************************************
References
******************************************

[1] Ansari, A.F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S.S., Arango, S.P., Kapoor, S. and Zschiegner, J., (2024).
Chronos: Learning the language of time series. *arXiv preprint*, `doi: 10.48550/arXiv.2403.07815 <https://doi.org/10.48550/arXiv.2403.07815>`__.