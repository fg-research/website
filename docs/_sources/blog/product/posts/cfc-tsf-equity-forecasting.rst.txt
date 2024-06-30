.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Forecasting Stock Returns with Liquid Neural Networks
   :keywords: Amazon SageMaker, Time Series, Liquid Neural Networks, Forecasting

######################################################################################
Forecasting Stock Returns with Liquid Neural Networks
######################################################################################

.. raw:: html

    <p>
    Stock return forecasting has been extensively studied by both academic researchers and
    industry practitioners. Numerous machine learning models have been proposed for this purpose,
    ranging from simple linear regressions to complex deep learning models <a href="#references">[1]</a>.
    In this post, we examine the performance of liquid neural networks <a href="#references">[4]</a>
    <a href="#references">[5]</a>, a new neural network architecture for sequential data.
    </p>

    <p>
    We will use our Amazon SageMaker implementation of liquid neural networks for probabilistic time series
    forecasting, the <a href="file:///Users/flaviagiammarino/website/docs/algorithms/time-series-forecasting/index.html#cfc-sagemaker-algorithm"
    target="_blank"> CfC SageMaker algorithm</a>. We will forecast the conditional mean and the
    conditional standard deviation of the 30-day returns of the S&P 500 using as input the S&P 500
    realized volatility as well as several implied volatility indices, similar to <a href="#references">[2]</a>.
    </p>

    <p>
    We will use the daily close prices from the 30<sup>th</sup> of June 2022 to
    the 29<sup>th</sup> of June 2024, which we will download using the <a href="https://github.com/ranaroussi/yfinance" target="_blank">Yahoo! Finance Python API</a>.
    We will train the model on the data up to the 8<sup>th</sup> of September 2023,
    and use the trained model to predict the subsequent data up to the 29<sup>th</sup> of June 2024.
    We will find that the CfC SageMaker algorithm achieves a mean absolute error of 1.4% and directional accuracy of 97.5%.
    </p>

******************************************
Model
******************************************

.. raw:: html

    <p>
    The closed-form continuous-depth network (CfC) is a new neural network architecture for
    sequential data <a href="#references">[4]</a>. CfCs belong to the class of continuous-time
    recurrent neural networks (CT-RNNs) <a href="#references">[3]</a>, where the evolution of
    the hidden state over time is described by an Ordinary Differential Equation (ODE).
    </p>

    <p>
    CfCs use the Liquid Time Constant (LTC) ODE <a href="#references">[5]</a>, where both the
    derivative and the time constant of the hidden state are determined by a neural network.
    Differently from other CT-RNNs (including LTCs), which use a numerical solver to find the
    ODE solution, CfCs use an approximate closed-form solution. As a results, CfCs achieve
    faster training and inference performance than other CT-RNNs.
    </p>

The hidden state :math:`x` of a CfC at time :math:`t` is given by

.. math::

    x(t) = \sigma(-f(x, I; \theta_f)t) \odot g(x, I; \theta_g) + [1 - \sigma(-[f(x, I; \theta_f)]t)] \odot h(x, I; \theta_h)

where :math:`\odot` is the Hadamard product, :math:`\sigma` is the sigmoid function, :math:`I`
is the input sequence, while :math:`f`, :math:`g` and :math:`h` are neural networks. The three
neural networks :math:`f`, :math:`g` and :math:`h` share a common backbone, which is a stack of
fully-connected layers with non-linear activation.

The backbone is followed by three separate neural network heads. The head of the :math:`g` and
:math:`h` neural networks is a fully-connected layer with hyperbolic tangent activation. The head
of the :math:`f` neural network is an affine function :math:`b + a(\Delta t)` where :math:`\Delta t`
is the time span (or time increment) between consecutive time steps, while the intercept :math:`b`
and slope :math:`a` are the outputs of two fully-connected layers with linear activation.

******************************************
Data
******************************************

==========================================
Outputs
==========================================

The model outputs are the 30-day returns of the S&P 500, which are calculated as follows

.. math::

    y(t) = \ln{P(t) / P(t-30)}

for each day :math:`t`, where `P(t)` is the close price of the S&P 500 on day :math:`t`.

==========================================
Inputs
==========================================

.. raw:: html

    <img
        id="cfc-tsf-forecasting-time-series"
        class="blog-post-image"
        alt="30-day returns, 30-day realized volatility and volatility indices from 2022-08-12 to 2024-06-29"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-forecasting/time_series_light.png
    />

    <p class="blog-post-image-caption">30-day returns, 30-day realized volatility and volatility indices from 2022-08-12 to 2024-06-29.</p>

******************************************
Code
******************************************

==========================================
Environment Set-Up
==========================================

We start by importing all the dependencies and setting up the SageMaker environment.

.. warning::

   To be able to run the code below, you need to have an active
   subscription to the CfC SageMaker algorithm. You can subscribe to a free trial from
   the `AWS Marketplace <https://aws.amazon.com/marketplace/pp/prodview-7s4giphluwgta>`__
   in order to get your Amazon Resource Name (ARN).
   In this post we use version 1.6 of the CfC SageMaker algorithm, which runs in the
   PyTorch 2.1.0 Python 3.10 deep learning container.

.. code:: python

    import io
    import sagemaker
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import yfinance as yf
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error, accuracy_score, f1_score

    # SageMaker session
    sagemaker_session = sagemaker.Session()

    # SageMaker role
    role = sagemaker.get_execution_role()

    # S3 bucket
    bucket = sagemaker_session.default_bucket()

    # EC2 instance
    instance_type = "ml.m5.4xlarge"


==========================================
Data Preparation
==========================================


==========================================
Training
==========================================

==========================================
Inference
==========================================

.. raw:: html

    <img
        id="cfc-tsf-forecasting-predictions"
        class="blog-post-image"
        alt="Actual and predicted 30-day returns from 2023-12-04 to 2024-06-28"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-forecasting/predictions_light.png
    />

    <p class="blog-post-image-caption">Actual and predicted 30-day returns from 2023-12-04 to 2024-06-28.</p>


.. raw:: html

    <img
        id="cfc-tsf-forecasting-forecasts"
        class="blog-post-image"
        alt="30-day returns forecasts from 2024-06-29 to 2024-07-28"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/equity-forecasting/forecasts_light.png
    />

    <p class="blog-post-image-caption">30-day returns forecasts from 2024-06-29 to 2024-07-28.</p>


******************************************
References
******************************************

[1] Kumbure, M.M., Lohrmann, C., Luukka, P. and Porras, J., (2022).
Machine learning techniques and data for stock market forecasting: A literature review.
*Expert Systems with Applications*, 197, p. 116659.
`doi: 10.1016/j.eswa.2022.116659 <https://doi.org/10.1016/j.eswa.2022.116659>`__.

[2] Campisi, G., Muzzioli, S. and De Baets, B., (2024).
A comparison of machine learning methods for predicting the direction of the US
stock market on the basis of volatility indices. *International Journal of Forecasting*, 40(3), pp. 869-880.
`doi: 10.1016/j.ijforecast.2023.07.002 <https://doi.org/10.1016/j.ijforecast.2023.07.002>`__.

[3] Funahashi, K.I. and Nakamura, Y., (1993). Approximation of dynamical systems by continuous
time recurrent neural networks. *Neural networks*, 6(6), pp.801-806.
`doi: 10.1016/S0893-6080(05)80125-X <https://doi.org/10.1016/S0893-6080(05)80125-X>`__.

[4] Hasani, R., Lechner, M., Amini, A., Liebenwein, L., Ray, A., Tschaikowski, M., Teschl, G. and Rus, D., (2022).
Closed-form continuous-time neural networks. *Nature Machine Intelligence*, 4(11), pp. 992-1003.
`doi: 10.1038/s42256-022-00556-7 <https://doi.org/10.1038/s42256-022-00556-7>`__.

[5] Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021).
Liquid time-constant networks. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(9), pp. 7657-7666.
`doi: 10.1609/aaai.v35i9.16936 <https://doi.org/10.1609/aaai.v35i9.16936>`__.
