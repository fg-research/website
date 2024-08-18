.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Predicting stock market trends with Amazon SageMaker Autopilot
   :keywords: Amazon SageMaker, Time Series, AutoML, Forecasting, Stock Market

######################################################################################
Predicting stock market trends with Amazon SageMaker Autopilot
######################################################################################

.. raw:: html

    <p>
    Building a well-performing machine learning model requires substantial time and resources.
    Automated machine learning (AutoML) automates the end-to-end process of building, training and tuning machine learning models.
    This not only accelerates the development cycle, but also makes machine learning more accessible
    to those without specialized data science expertise.
    </p>

    <p>
    In this post, we use <a href=https://aws.amazon.com/sagemaker/autopilot target=_blank>
    Amazon SageMaker Autopilot</a> for building a stock market trend prediction model.
    We will run an AutoML job for optimizing an XGBoost classifier to predict the
    direction of the S&P500 (up or down) one day ahead using as input a set of technical indicators.
    </p>

    <p>
    We will download the daily prices of the S&P500 from the 1<sup>st</sup> of August 2021 to
    the 31<sup>st</sup> of July 2024 from <a href="https://finance.yahoo.com" target="_blank">Yahoo! Finance</a>.
    We will train the model on the data up to the 3<sup>rd</sup> of May 2024, and validate the model on
    the subsequent 30 days of data up to the 17<sup>th</sup> of June 2024. We will then test the
    identified best model on the remaining 30 days of data up to the 31<sup>st</sup> of July 2024.
    We will find that the XGBoost model achieves a mean directional accuracy of %
    over the considered 30-days period.
    </p>

******************************************
References
******************************************

[1] Kara, Y., Boyacioglu, M. A., & Baykan, Ö. K. (2011).
Predicting direction of stock price index movement using artificial neural networks and support vector machines:
The sample of the Istanbul Stock Exchange. *Expert Systems with Applications*, 38(5), 5311-5319.
`doi: doi:10.1016/j.eswa.2010.10.027 <https://doi.org/doi:10.1016/j.eswa.2010.10.027>`__.

