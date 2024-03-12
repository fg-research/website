.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: ECG anomaly detection with the LSTM-AD SageMaker Algorithm
   :keywords: Amazon SageMaker, Time Series, LSTM, Anomaly Detection

######################################################################################
ECG anomaly detection with the LSTM-AD SageMaker Algorithm
######################################################################################

.. raw:: html

    <p>
    </p>

******************************************
Model
******************************************
.. raw:: html

    <p>
    The LSTM-AD model predicts the time series with a multivariate stacked LSTM model.
    </p>

    <p>
    The model parameters are learned on a training set containing only normal data (i.e. without anomalies)
    by minimizing the Mean Squared Error (MSE) between the actual and predicted values of the time series.
    </p>

    <p>
    After the model has been trained, a multivariate Gaussian distribution is fitted to the model’s prediction errors
    on an independent validation set (also without anomalies) using Maximum Likelihood Estimation (MLE).
    </p>

    <p>
    At inference time, the model predicts the values of all the time series (which can now include anomalies)
    at each time step, and calculates the likelihood of the model’s prediction errors under the fitted multivariate
    Gaussian distribution.
    </p>

    <p>
    The computed Gaussian likelihood is then used as a normality score: the lower the Gaussian
    likelihood at a given a time step, the more likely the time step is to be an anomaly.
    </p>

    <p>
    If enough labelled anomalous data is available, an optimal threshold on the normality score can be determined
    by maximizing the F-beta score between the actual and predicted anomaly labels.
    </p>

******************************************
Data
******************************************
.. raw:: html

    <p>
    We use dataset number 179 from the Hexagon ML / UCR Time Series Anomaly Detection Archive.
    The dataset includes a single time series representing a human subject ECG trace sourced from
    record s30791 in the Long Term ST Database (LTST DB). The length of the time series is 55000
    observations. The first 23000 observations are included in the training set, while the remaining
    32000 observations are included in the test set. The training set contains only normal data,
    while the test set contains an anomalous heartbeat between observations 52600 and 52800.
    </p>