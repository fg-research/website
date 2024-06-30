.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Forecasting Stock Returns with Liquid Neural Networks
   :keywords: Amazon SageMaker, Time Series, Liquid Neural Networks, Forecasting

######################################################################################
Forecasting Stock Returns with Liquid Neural Networks
######################################################################################




******************************************
Model
******************************************

******************************************
Data
******************************************

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

[3] Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021).
Liquid time-constant networks. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(9), pp. 7657-7666.
`doi: 10.1609/aaai.v35i9.16936 <https://doi.org/10.1609/aaai.v35i9.16936>`__.

[4] Hasani, R., Lechner, M., Amini, A., Liebenwein, L., Ray, A., Tschaikowski, M., Teschl, G. and Rus, D., (2022).
Closed-form continuous-time neural networks. *Nature Machine Intelligence*, 4(11), pp. 992-1003.
`doi: 10.1038/s42256-022-00556-7 <https://doi.org/10.1038/s42256-022-00556-7>`__.