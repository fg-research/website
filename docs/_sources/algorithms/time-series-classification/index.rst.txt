.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Perform time series classification in Amazon SageMaker
   :keywords: Amazon SageMaker, Time Series, Classification

.. _time-series-classification-algorithms:

########################################################
Time Series Classification
########################################################

.. rst-class:: lead

   Perform time series classification in Amazon SageMaker

.. table::
   :width: 100%

   ===========================================================  ======================================== ======================================== ============================================ ================================================
   SageMaker Algorithm                                          CPU Training                             GPU Training                             Multi-GPU Training                           Incremental Training
   ===========================================================  ======================================== ======================================== ============================================ ================================================
   :ref:`LSTM-FCN <lstm-fcn-sagemaker-algorithm>`               :octicon:`check;1rem;check-icon`         :octicon:`check;1rem;check-icon`         :octicon:`check;1rem;check-icon`             :octicon:`check;1rem;check-icon`
   :ref:`InceptionTime <inception-time-sagemaker-algorithm>`    :octicon:`check;1rem;check-icon`         :octicon:`check;1rem;check-icon`         :octicon:`check;1rem;check-icon`             :octicon:`check;1rem;check-icon`
   :ref:`CNN-SVC <cnn-svc-sagemaker-algorithm>`                 :octicon:`check;1rem;check-icon`         :octicon:`check;1rem;check-icon`         :octicon:`x;1rem;x-icon`                     :octicon:`x;1rem;x-icon`
   ===========================================================  ======================================== ======================================== ============================================ ================================================

.. _lstm-fcn-sagemaker-algorithm:

******************************************
LSTM-FCN SageMaker Algorithm
******************************************
.. raw:: html

    <p>
        The LSTM-FCN SageMaker Algorithm performs time series classification with the <a href="https://arxiv.org/pdf/1709.05206.pdf" target="_blank">Long Short-Term Memory Fully Convolutional Network (LSTM-FCN)</a>.
        The LSTM-FCN model consists of two blocks: a recurrent block and a convolutional block.
        The two blocks process the input time series in parallel.
        After that their output representations are concatenated and passed to a final output layer with softmax activation.
        For additional information, see the algorithm's
        <a href="https://aws.amazon.com/marketplace/pp/prodview-vzxmyw25oqtx6" target="_blank">AWS Marketplace</a>
        listing page and
        <a href="https://github.com/fg-research/lstm-fcn-sagemaker" target="_blank">GitHub</a>
        repository.
    </p>

.. _inception-time-sagemaker-algorithm:

******************************************
InceptionTime SageMaker Algorithm
******************************************
.. raw:: html

    <p>
        The InceptionTime SageMaker Algorithm performs time series classification with the <a href="https://arxiv.org/pdf/1909.04939.pdf" target="_blank">InceptionTime Network</a>.
        The InceptionTime network consists of a stack of Inception blocks linked by residual connections.
        Each block contains three convolutional layers and a max pooling layer.
        The four layers process the block input in parallel.
        After that their output representations are concatenated and passed to a batch normalization layer followed by a dense layer with ReLU activation.
        The output of the last block is passed to an average pooling layer, and then to a final output layer with softmax activation.
        The algorithm trains an ensemble of InceptionTime networks and generates the final predicted class labels by averaging the class probabilities predicted by the different networks in the ensemble.
        For additional information, see the algorithm's
        <a href="https://aws.amazon.com/marketplace/pp/prodview-omz7rumnllmla" target="_blank">AWS Marketplace</a>
        listing page and
        <a href="https://github.com/fg-research/inception-time-sagemaker" target="_blank">GitHub</a>
        repository.
    </p>

.. _cnn-svc-sagemaker-algorithm:

******************************************
CNN-SVC SageMaker Algorithm
******************************************
.. raw:: html

    <p>
        The CNN-SVC SageMaker Algorithm performs time series classification with an <a href="https://arxiv.org/pdf/1901.10738.pdf" target="_blank">unsupervised convolutional neural network (CNN)</a> followed by a support vector classifier (SVC).
        The CNN network encodes the input time series into a number of time-independent features, which are then used as inputs by the SVC algorithm.
        The CNN network consists of a stack of exponentially dilated causal convolutional blocks with residual connections, and
        is trained in an unsupervised manner using a contrastive learning procedure that minimizes the triplet loss.
        The algorithm can be used for time series with different lengths and with missing values.
        The algorithm also supports missing class labels.
        For additional information, see the algorithm's
        <a href="https://aws.amazon.com/marketplace/pp/prodview-mo7cf4nrgrbxk" target="_blank">AWS Marketplace</a>
        listing page and
        <a href="https://github.com/fg-research/cnn-svc-sagemaker" target="_blank">GitHub</a>
        repository.
    </p>
