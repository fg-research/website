.. _time-series-forecasting-algorithms:

########################################################
Time Series Forecasting
########################################################

.. rst-class:: lead

   Perform time series forecasting in Amazon SageMaker.

.. table::
   :width: 100%

   ============================================  ======================================== ======================================== ============================================ ================================================
   SageMaker Algorithm                           CPU Training                             GPU Training                             Multi-GPU Training                           Incremental Training
   ============================================  ======================================== ======================================== ============================================ ================================================
   :ref:`CfC <cfc-sagemaker-algorithm>`          :octicon:`check;1rem;check-icon`         :octicon:`check;1rem;check-icon`          :octicon:`check;1rem;check-icon`             :octicon:`check;1rem;check-icon`
   ============================================  ======================================== ======================================== ============================================ ================================================

.. _cfc-sagemaker-algorithm:

******************************************
CfC SageMaker Algorithm
******************************************
The CfC SageMaker Algorithm performs time series forecasting with the `Closed-Form Continuous-Depth (CfC) network <https://doi.org/10.1038/s42256-022-00556-7>`_.
CfCs are continuous-time recurrent neural networks implementing an approximate closed-form solution of the `Liquid Time Constant (LTC) <https://doi.org/10.1609/aaai.v35i9.16936>`_ ordinary differential equation.
CfCs are also referred to as `Liquid Neural Networks <https://news.mit.edu/2021/machine-learning-adapts-0128>`_.
CfCs are particularly suitable for irregularly-sampled time series, that is for time series whose values are not observed at a constant frequency.
The algorithm can be used for both univariate and multivariate time series and supports the inclusion of external features.
[`AWS Marketplace <https://aws.amazon.com/marketplace/pp/prodview-7s4giphluwgta>`__]
[`GitHub <https://github.com/fg-research/cfc-tsf-sagemaker>`__]
