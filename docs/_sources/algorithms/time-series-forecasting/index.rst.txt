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
        The LNN SageMaker Algorithm performs time series forecasting with Liquid Neural Networks (LNNs).
        LNNs belong to the class of continuous-time recurrent neural networks (CT-RNNs), where the evolution
        of the hidden state over time is described by an Ordinary Differential Equation (ODE).
        The algorithm uses the <a href="https://doi.org/10.1038/s42256-022-00556-7" target="_blank">closed-form continuous-depth (CfC) network</a>
        implementation of LNNs.
        Differently from other CT-RNNs, including other LNNs such as <a href="https://doi.org/10.1609/aaai.v35i9.16936" target="_blank">liquid time-constant (LTC) networks</a>,
        which use a numerical solver to find the ODE solution, CfCs use an approximate closed-form solution.
        As a results, CfCs achieve faster training and inference performance than other LNNs and CT-RNNs.
        The algorithm can be used for both univariate and multivariate time series and supports the inclusion of external features.
        For additional information, see the algorithm's
        <a href="https://aws.amazon.com/marketplace/pp/prodview-7s4giphluwgta" target="_blank">AWS Marketplace</a>
        listing page and
        <a href="https://github.com/fg-research/lnn-sagemaker" target="_blank">GitHub</a>
        repository.
    </p>


