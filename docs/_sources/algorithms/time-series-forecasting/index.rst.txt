.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Perform time series forecasting in Amazon SageMaker
   :keywords: Amazon SageMaker, Time Series, Forecasting

.. _time-series-forecasting-algorithms:

########################################################
Time Series Forecasting
########################################################

.. rst-class:: lead

   Perform time series forecasting in Amazon SageMaker

.. table::
   :width: 100%

   ============================================  ======================================== ======================================== ============================================ ================================================
   SageMaker Algorithm                           CPU Training                             GPU Training                             Multi-GPU Training                           Incremental Training
   ============================================  ======================================== ======================================== ============================================ ================================================
   :ref:`RNN <rnn-sagemaker-algorithm>`          :octicon:`check;1rem;check-icon`         :octicon:`check;1rem;check-icon`          :octicon:`check;1rem;check-icon`             :octicon:`check;1rem;check-icon`
   :ref:`LNN <lnn-sagemaker-algorithm>`          :octicon:`check;1rem;check-icon`         :octicon:`check;1rem;check-icon`          :octicon:`check;1rem;check-icon`             :octicon:`check;1rem;check-icon`
   ============================================  ======================================== ======================================== ============================================ ================================================

.. _rnn-sagemaker-algorithm:

******************************************
RNN SageMaker Algorithm
******************************************
.. raw:: html

    <p>
        The RNN SageMaker Algorithm performs time series forecasting with Recurrent Neural Networks (RNNs).
        The model consists of a stack of RNN layers with either LSTM or GRU cells.
        Each RNN layer is followed by an activation layer and a dropout layer.
        The model is trained by minimizing the negative Gaussian log-likelihood and outputs the predicted mean
        and standard deviation at each future time step.
        The algorithm can be used for both univariate and multivariate time series and supports the inclusion of external features.
        For additional information, see the algorithm's
        <a href="https://aws.amazon.com/marketplace/pp/prodview-p5cr7ncmdcziw" target="_blank">AWS Marketplace</a>
        listing page and
        <a href="https://github.com/fg-research/rnn-sagemaker" target="_blank">GitHub</a>
        repository.
    </p>

.. _lnn-sagemaker-algorithm:

******************************************
LNN SageMaker Algorithm
******************************************
.. raw:: html

    <p>
        The LNN SageMaker Algorithm performs time series forecasting with Liquid Neural Networks (LNNs).
        The algorithm uses the <a href="https://doi.org/10.1038/s42256-022-00556-7" target="_blank">closed-form continuous-depth (CfC) network</a>
        implementation of LNNs, which implement an approximate closed-form solution of the Liquid Time Constant (LTC) Ordinary Differential Equation (ODE).
        The algorithm can be used for both univariate and multivariate time series and supports the inclusion of external features.
        For additional information, see the algorithm's
        <a href="https://aws.amazon.com/marketplace/pp/prodview-7s4giphluwgta" target="_blank">AWS Marketplace</a>
        listing page and
        <a href="https://github.com/fg-research/lnn-sagemaker" target="_blank">GitHub</a>
        repository.
    </p>


