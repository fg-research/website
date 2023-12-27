.. meta::
   :description: Time series clustering in Amazon SageMaker.

########################################################
Time Series Clustering
########################################################

.. rst-class:: lead

   Perform time series clustering in Amazon SageMaker.

.. table::
   :width: 100%

   ===================================================  ======================================== ======================================== ============================================ ================================================
   SageMaker Algorithm                                  CPU Training                             GPU Training                             Multi-GPU Training                           Incremental Training
   ===================================================  ======================================== ======================================== ============================================ ================================================
   :ref:`CNN-KMeans <cnn-kmeans-sagemaker-algorithm>`   :octicon:`check;1.1em;check-icon`        :octicon:`check;1.1em;check-icon`        :octicon:`x;1.1em;x-icon`                    :octicon:`x;1.1em;x-icon`
   ===================================================  ======================================== ======================================== ============================================ ================================================

******************************************
CNN-KMeans SageMaker Algorithm
******************************************
.. _cnn-kmeans-sagemaker-algorithm:

The CNN-KMeans SageMaker Algorithm performs time series clustering with an `unsupervised convolutional neural network (CNN) <https://arxiv.org/pdf/1901.10738.pdf>`_ followed by a K-Means clusterer.
The CNN network encodes the input time series into a number of time-independent features, which are then used as input by the K-Means algorithm.
The CNN network consists of a stack of exponentially dilated causal convolutional blocks with residual connections
and is trained in an unsupervised manner using a contrastive learning procedure that minimizes the triplet loss.
The algorithm can be used for time series with different lengths or with missing values.
[`AWS Marketplace <https://aws.amazon.com/marketplace/pp/prodview-3hdblqdz5nx4m>`_]
[`GitHub <https://github.com/fg-research/cnn-kmeans-sagemaker>`_]


