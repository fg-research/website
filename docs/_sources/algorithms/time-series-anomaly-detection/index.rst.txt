.. meta::
   :description: Time series anomaly detection in Amazon SageMaker.

########################################################
Time Series Anomaly Detection
########################################################

.. rst-class:: lead

   Perform time series anomaly detection in Amazon SageMaker.

.. table::
   :width: 100%

   ================================================  ======================================== ======================================== ============================================ ================================================
   SageMaker Algorithm                               CPU Training                             GPU Training                             Multi-GPU Training                           Incremental Training
   ================================================  ======================================== ======================================== ============================================ ================================================
   :ref:`LSTM-AD <lstm-ad-sagemaker-algorithm>`      :octicon:`check;1.1em;check-icon`        :octicon:`check;1.1em;check-icon`        :octicon:`check;1.1em;check-icon`            :octicon:`x;1.1em;x-icon`
   :ref:`LSTM-AE <lstm-ae-sagemaker-algorithm>`      :octicon:`check;1.1em;check-icon`        :octicon:`check;1.1em;check-icon`        :octicon:`check;1.1em;check-icon`            :octicon:`x;1.1em;x-icon`
   ================================================  ======================================== ======================================== ============================================ ================================================

******************************************
LSTM-AD SageMaker Algorithm
******************************************
.. _lstm-ad-sagemaker-algorithm:

The LSTM-AD SageMaker Algorithm performs time series anomaly detection with the `Long Short-Term Memory Network for Anomaly Detection (LSTM-AD) <https://www.esann.org/sites/default/files/proceedings/legacy/es2015-56.pdf>`_.
The LSTM-AD model predicts the future values of the time series with a stacked LSTM network.
After that it fits a Gaussian distribution to the prediction errors obtained on normal data.
The Gaussian likelihood of the prediction errors is then used as a normality score.
The lower the Gaussian likelihood at a given a time step, the more likely the time step is to be an anomaly.
[`AWS Marketplace <https://aws.amazon.com/marketplace/pp/prodview-4pbvedtnnlphw>`__]
[`GitHub <https://github.com/fg-research/lstm-ad-sagemaker>`__]

******************************************
LSTM-AE SageMaker Algorithm
******************************************
.. _lstm-ae-sagemaker-algorithm:

The LSTM-AE SageMaker Algorithm performs time series anomaly detection with the `Long Short Term Memory Networks based Encoder-Decoder scheme for Anomaly Detection (LSTM-AE) <https://arxiv.org/pdf/1607.00148.pdf>`_.
The LSTM-AE model reconstructs the observed values of the time series with an LSTM autoencoder.
After that it fits a Gaussian distribution to the reconstruction errors obtained on normal data.
The squared Mahalanobis distance between the reconstruction errors and the fitted Gaussian distribution is then used as an anomaly score.
The larger the squared Mahalanobis distance at a given a time step, the more likely the time step is to be an anomaly.
[`AWS Marketplace <https://aws.amazon.com/marketplace/pp/prodview-up2haipz3j472>`__]
[`GitHub <https://github.com/fg-research/lstm-ae-sagemaker>`__]

