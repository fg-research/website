.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Forecasting Stock Returns with Liquid Neural Networks
   :keywords: Amazon SageMaker, Time Series, Recurrent Neural Networks, Forecasting, Forex Market

############################################################################################################
Forecasting exchange rates with long short-term memory (LSTM) networks using the RNN SageMaker Algorithm
############################################################################################################

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
