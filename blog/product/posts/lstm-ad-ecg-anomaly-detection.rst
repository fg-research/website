.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: ECG anomaly detection with the LSTM-AD SageMaker Algorithm
   :keywords: Amazon SageMaker, Time Series, LSTM, Anomaly Detection

######################################################################################
ECG anomaly detection with the LSTM-AD SageMaker Algorithm
######################################################################################

.. raw:: html

    <p>
    Anomaly detection in Electrocardiogram (ECG) signals is crucial for early diagnosis
    and treatment of cardiovascular diseases. Machine learning algorithms can automate
    the process of ECG analysis, reducing the need for manual inspection by healthcare
    professionals.
    </p>

    <p>
    Different machine learning algorithms have been studied in the literature on ECG
    anomaly detection.
    </p>

    <p>
    In this post, we will focus on the <a href="https://www.esann.org/sites/default/files/proceedings/legacy/es2015-56.pdf"
    target="_blank"> Long Short-Term Memory Network for Anomaly Detection</a>
    <a href="#references">[2]</a>, which we will refer to as the LSTM-AD model.
    We will demonstrate how to use our Amazon SageMaker implementation of the LSTM-AD model,
    the <a href="https://fg-research.com/algorithms/time-series-anomaly-detection/index.html#lstm-ad-sagemaker-algorithm"
    target="_blank">LSTM-AD SageMaker algorithm</a>, for detecting anomalies in an ECG trace.
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

.. raw:: html

    <img
        id="lstm-ad-ecg-anomaly-detection-dataset"
        class="blog-post-image"
        alt="Hexagon ML / UCR dataset N°179 (combined training and test sets)"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/ecg-anomaly-detection/data_light.png
    />

    <p class="blog-post-image-caption">Hexagon ML / UCR dataset N°179 (combined training and test sets).</p>



.. raw:: html

    <img
        id="lstm-ad-ecg-anomaly-detection-results"
        class="blog-post-image"
        alt="Results on Hexagon ML / UCR dataset №179 (test set)"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/ecg-anomaly-detection/results_light.png
    />

    <p class="blog-post-image-caption">Results on Hexagon ML / UCR dataset №179 (test set).</p>