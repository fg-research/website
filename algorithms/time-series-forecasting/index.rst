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
   :ref:`LNN <lnn-sagemaker-algorithm>`          :octicon:`check;1rem;check-icon`         :octicon:`check;1rem;check-icon`          :octicon:`check;1rem;check-icon`             :octicon:`check;1rem;check-icon`
   ============================================  ======================================== ======================================== ============================================ ================================================

.. _lnn-sagemaker-algorithm:

******************************************
LNN SageMaker Algorithm
******************************************
.. raw:: html

    <p>
        The LNN SageMaker Algorithm performs time series forecasting with <a href="https://news.mit.edu/2021/machine-learning-adapts-0128" target="_blank">Liquid Neural Networks (LNNs)</a>.
        LNNs are continuous-time recurrent neural networks which implement an approximate closed-form solution of the <a href="https://doi.org/10.1609/aaai.v35i9.16936" target="_blank">Liquid Time Constant (LTC)</a> ordinary differential equation.
        The algorithm can be used for both univariate and multivariate time series and supports the inclusion of external features.
        For additional information, see the algorithm's
        <a href="https://aws.amazon.com/marketplace/pp/prodview-7s4giphluwgta" target="_blank">AWS Marketplace</a>
        listing page and
        <a href="https://github.com/fg-research/lnn-sagemaker" target="_blank">GitHub</a>
        repository.
    </p>


