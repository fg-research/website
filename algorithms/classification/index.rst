.. meta::
   :description: Time series classification in Amazon SageMaker.

########################################################
Time Series Classification
########################################################

.. rst-class:: lead

   Perform time series classification in Amazon SageMaker.


LSTM-FCN SageMaker Algorithm
=============================================
The LSTM-FCN SageMaker Algorithm performs time series classification with the `Long Short-Term Memory Fully Convolutional Network (LSTM-FCN) <https://doi.org/10.1109/ACCESS.2017.2779939>`_.
The LSTM-FCN model consists of two blocks: a recurrent block and a convolutional block.
The two blocks process the input time series in parallel.
After that their output representations are concatenated and passed to a final output layer with softmax activation.
[`AWS Marketplace <https://aws.amazon.com/marketplace/pp/prodview-vzxmyw25oqtx6>`_]
[`GitHub <https://github.com/fg-research/lstm-fcn-sagemaker>`_]

InceptionTime SageMaker Algorithm
=============================================
The InceptionTime SageMaker Algorithm performs time series classification with the `InceptionTime Network <https://doi.org/10.1007/s10618-020-00710-y>`_.
The InceptionTime network consists of a stack of Inception blocks linked by residual connections.
Each block contains three convolutional layers and a max pooling layer.
The four layers process the block input in parallel.
After that their output representations are concatenated and passed to a batch normalization layer followed by a dense layer with ReLU activation.
The output of the last block is passed to an average pooling layer, and then to a final output layer with softmax activation.
The algorithm trains an ensemble of InceptionTime networks and derives the final predicted class labels by averaging the class probabilities predicted by the different networks in the ensemble.
[`AWS Marketplace <https://aws.amazon.com/marketplace/pp/prodview-omz7rumnllmla>`_]
[`GitHub <https://github.com/fg-research/inception-time-sagemaker>`_]

CNN-SVC SageMaker Algorithm
=============================================
The CNN-SVC SageMaker Algorithm performs time series classification with an `unsupervised convolutional neural network (CNN) <https://doi.org/10.48550/arXiv.1901.10738>`_ followed by a support vector classifier (SVC).
The CNN network encodes the input time series into a number of time-independent features, which are then used as inputs by the SVC algorithm.
The CNN network consists of a stack of exponentially dilated causal convolutional blocks with residual connections.
The CNN network is trained in an unsupervised manner using a contrastive learning procedure that minimizes the triplet loss.
The algorithm can be used for time series with different lengths or with missing values.
The algorithm also supports missing class labels.
[`AWS Marketplace <https://aws.amazon.com/marketplace/pp/prodview-mo7cf4nrgrbxk>`_]
[`GitHub <https://github.com/fg-research/cnn-svc-sagemaker>`_]

