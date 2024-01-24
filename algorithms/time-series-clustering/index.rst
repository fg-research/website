.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Perform time series clustering in Amazon SageMaker
   :keywords: Amazon SageMaker, Time Series, Clustering

.. _time-series-clustering-algorithms:

########################################################
Time Series Clustering
########################################################

.. rst-class:: lead

   Perform time series clustering in Amazon SageMaker

.. table::
   :width: 100%

   ===================================================  ======================================== ======================================== ============================================ ================================================
   SageMaker Algorithm                                  CPU Training                             GPU Training                             Multi-GPU Training                           Incremental Training
   ===================================================  ======================================== ======================================== ============================================ ================================================
   :ref:`CNN-KMeans <cnn-kmeans-sagemaker-algorithm>`   :octicon:`check;1rem;check-icon`         :octicon:`check;1rem;check-icon`         :octicon:`x;1rem;x-icon`                     :octicon:`x;1rem;x-icon`
   ===================================================  ======================================== ======================================== ============================================ ================================================

.. _cnn-kmeans-sagemaker-algorithm:

******************************************
CNN-KMeans SageMaker Algorithm
******************************************
.. raw:: html

    <p>
        The CNN-KMeans SageMaker Algorithm performs time series clustering with an <a href="https://arxiv.org/pdf/1901.10738.pdf" target="_blank">unsupervised convolutional neural network (CNN)</a> followed by a K-Means clusterer.
        The CNN network encodes the input time series into a number of time-independent features, which are then used as input by the K-Means algorithm.
        The CNN network consists of a stack of exponentially dilated causal convolutional blocks with residual connections,
        and is trained in an unsupervised manner using a contrastive learning procedure that minimizes the triplet loss.
        The algorithm can be used for time series with different lengths and with missing values.
        For additional information, see the algorithm's
        <a href="https://aws.amazon.com/marketplace/pp/prodview-3hdblqdz5nx4m" target="_blank">AWS Marketplace</a>
        listing page and
        <a href="https://github.com/fg-research/cnn-kmeans-sagemaker" target="_blank">GitHub</a>
        repository.
    </p>



