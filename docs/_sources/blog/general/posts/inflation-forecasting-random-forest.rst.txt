.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Forecasting US inflation with random forests
   :keywords: Time Series, Forecasting, Machine Learning, Macroeconomics, Inflation

######################################################################################
Forecasting US inflation with random forests
######################################################################################

.. raw:: html

    <p>
    Inflation forecasts are used for informing economic decisions at various levels,
    from households to businesses and policymakers. Machine learning approaches
    offer several advantages for inflation forecasting, including the ability to
    handle large and complex datasets, capture nonlinear relationships, and adapt
    to changing economic conditions.
    </p>

    <p>
    Several recent papers have studied the problem of forecasting US inflation with
    machine learning methods using the <a href="https://research.stlouisfed.org/econ/mccracken/fred-databases/" target="_blank">FRED-MD</a>
    dataset <a href="#references">[1]</a>. The FRED-MD dataset includes
    over 100 monthly time series of 8 different categories of US economic indicators:
    output and income, labour market, consumption and orders, orders and inventory,
    money and credit, interest rates and exchange rates, prices, and stock market.
    </p>

    <p>
    In this post, we will focus on the random forest model introduced in <a href="#references">[2]</a>,
    which was found to outperform several


    </p>

******************************************
Model
******************************************

******************************************
Data
******************************************
.. raw:: html

    For a detailed overview of the FRED-MD dataset we refer to
    <a href=https://fg-research.com/blog/general/posts/fred-md-overview.html
    target="_blank">our previous blog post</a>.

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
Hyperparameter Tuning
==========================================

==========================================
Training
==========================================

==========================================
Inference
==========================================

******************************************
References
******************************************

[1] McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for macroeconomic research. *Journal of Business & Economic Statistics*, 34(4), 574-589. `doi: 10.1080/07350015.2015.1086655 <https://doi.org/10.1080/07350015.2015.1086655>`__.

[2] Medeiros, M. C., Vasconcelos, G. F., Veiga, Á., & Zilberman, E. (2021). Forecasting inflation in a data-rich environment: the benefits of machine learning methods. *Journal of Business & Economic Statistics*, 39(1), 98-119. `doi: 10.1080/07350015.2019.1637745 <https://doi.org/10.1080/07350015.2019.1637745>`__.
