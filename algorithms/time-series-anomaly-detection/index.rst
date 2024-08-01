.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Perform time series anomaly detection in Amazon SageMaker
   :keywords: Amazon SageMaker, Time Series, Anomaly Detection

.. _time-series-anomaly-detection-algorithms:

########################################################
Time Series Anomaly Detection
########################################################

.. rst-class:: lead

   Perform time series anomaly detection in Amazon SageMaker

.. table::
   :width: 100%

   ================================================  ======================================== ======================================== ============================================ ================================================
   SageMaker Algorithm                               CPU Training                             GPU Training                             Multi-GPU Training                           Incremental Training
   ================================================  ======================================== ======================================== ============================================ ================================================
   :ref:`LSTM-AD <lstm-ad-sagemaker-algorithm>`      :octicon:`check;1rem;check-icon`         :octicon:`check;1rem;check-icon`         :octicon:`check;1rem;check-icon`             :octicon:`x;1rem;x-icon`
   :ref:`LSTM-AE <lstm-ae-sagemaker-algorithm>`      :octicon:`check;1rem;check-icon`         :octicon:`check;1rem;check-icon`         :octicon:`check;1rem;check-icon`             :octicon:`x;1rem;x-icon`
   ================================================  ======================================== ======================================== ============================================ ================================================

.. _lstm-ad-sagemaker-algorithm:

******************************************
LSTM-AD SageMaker Algorithm
******************************************
.. raw:: html

    <p>
        The LSTM-AD SageMaker Algorithm performs time series anomaly detection with the <a href="https://www.esann.org/sites/default/files/proceedings/legacy/es2015-56.pdf" target="_blank">Long Short-Term Memory Network for Anomaly Detection (LSTM-AD)</a>.
        The LSTM-AD model predicts the future values of the time series with a stacked LSTM network.
        After that it fits a Gaussian distribution to the prediction errors obtained on normal data.
        The Gaussian likelihood of the prediction errors is then used as a normality score.
        The lower the Gaussian likelihood at a given a time step, the more likely the time step is to be an anomaly.
        The algorithm can be used for both univariate and multivariate time series.
        For additional information, see the algorithm's
        <a href="https://aws.amazon.com/marketplace/pp/prodview-4pbvedtnnlphw" target="_blank">AWS Marketplace</a>
        listing page and
        <a href="https://github.com/fg-research/lstm-ad-sagemaker" target="_blank">GitHub</a>
        repository.
    </p>

.. _lstm-ae-sagemaker-algorithm:

******************************************
LSTM-AE SageMaker Algorithm
******************************************
.. raw:: html

    <p>
        The LSTM-AE SageMaker Algorithm performs time series anomaly detection with the <a href="https://arxiv.org/pdf/1607.00148.pdf" target="_blank">Long Short Term Memory Networks based Encoder-Decoder scheme for Anomaly Detection (LSTM-AE)</a>.
        The LSTM-AE model reconstructs the observed values of the time series with an LSTM autoencoder.
        After that it fits a Gaussian distribution to the reconstruction errors obtained on normal data.
        The squared Mahalanobis distance between the reconstruction errors and the fitted Gaussian distribution is then used as an anomaly score.
        The larger the squared Mahalanobis distance at a given a time step, the more likely the time step is to be an anomaly.
        The algorithm can be used for both univariate and multivariate time series.
        For additional information, see the algorithm's
        <a href="https://aws.amazon.com/marketplace/pp/prodview-up2haipz3j472" target="_blank">AWS Marketplace</a>
        listing page and
        <a href="https://github.com/fg-research/lstm-ae-sagemaker" target="_blank">GitHub</a>
        repository.
    </p>


.. raw:: html

    <p style="margin-bottom: 1rem"> <br/> </p>

------

.. grid:: 3

    .. grid-item::
        :columns: 5

        .. toctree::
           :caption: Algorithms
           :maxdepth: 1

           algorithms/time-series-forecasting/index
           algorithms/time-series-anomaly-detection/index
           algorithms/time-series-classification/index
           algorithms/time-series-clustering/index

    .. grid-item::
        :columns: 3

        .. toctree::
           :caption: Blog
           :maxdepth: 1

           blog/product/index
           blog/general/index

    .. grid-item::
        :columns: 4

        .. toctree::
           :caption: Terms and Conditions
           :maxdepth: 1

           terms/disclaimer/index
           terms/eula/index
