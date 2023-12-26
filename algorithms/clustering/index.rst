.. meta::
   :description: Time series clustering in Amazon SageMaker.

########################################################
Time Series Clustering
########################################################

.. rst-class:: lead

   Perform time series clustering in Amazon SageMaker.

CNN-KMeans SageMaker Algorithm
=============================================
The CNN-KMeans SageMaker Algorithm performs time series clustering with an `unsupervised convolutional neural network (CNN) <https://doi.org/10.48550/arXiv.1901.10738>`_ followed by a K-Means clusterer.
The CNN network encodes the input time series into a number of time-independent features, which are then used as input by the K-Means algorithm.
The CNN network consists of a stack of exponentially dilated causal convolutional blocks with residual connections and is trained in an unsupervised manner using contrastive learning.
The algorithm can be used for time series with different lengths or with missing values.
[`AWS Marketplace <https://aws.amazon.com/marketplace/pp/prodview-3hdblqdz5nx4m>`_]
[`GitHub <https://github.com/fg-research/cnn-kmeans-sagemaker>`_]


