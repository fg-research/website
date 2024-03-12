.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: ECG anomaly detection with the LSTM-AD SageMaker Algorithm
   :keywords: Amazon SageMaker, Time Series, LSTM, Anomaly Detection

######################################################################################
ECG anomaly detection with the LSTM-AD SageMaker Algorithm
######################################################################################

.. raw:: html

    <p>
    Anomaly detection in electrocardiogram (ECG) signals is crucial for early diagnosis
    and treatment of cardiovascular diseases. With the development of wearable ECG sensors,
    it has become possible to monitor a patient's heart condition continuously and in real
    time. However, it is impracticable for healthcare professional to manually review such
    a large amount of ECG readings. Machine learning algorithms can automate the process of
    ECG analysis, reducing the need for manual inspection by healthcare professionals.
    </p>

    <p>
    Different supervised <a href="#references">[1]</a> and unsupervised <a href="#references">[2]</a>
    machine learning algorithms have been studied in the literature on ECG anomaly detection.
    In this post, we will focus on the
    <a href="https://www.esann.org/sites/default/files/proceedings/legacy/es2015-56.pdf" target="_blank">
    Long Short-Term Memory Network for Anomaly Detection</a> <a href="#references">[3]</a>,
    which we will refer to as the LSTM-AD model. We will demonstrate how to use our
    Amazon SageMaker implementation of the LSTM-AD model, the
    <a href="https://fg-research.com/algorithms/time-series-anomaly-detection/index.html#lstm-ad-sagemaker-algorithm"
    target="_blank">LSTM-AD SageMaker algorithm</a>,
    for detecting anomalies in an ECG trace.
    </p>

******************************************
Model
******************************************
.. raw:: html

    <p>
    The LSTM-AD model predicts the future values of the time series with a multivariate stacked LSTM model.
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
    Gaussian distribution. The computed Gaussian likelihood is then used as a normality score: the lower the Gaussian
    likelihood at a given a time step, the more likely the time step is to be an anomaly.
    </p>

.. note::

    If enough labelled anomalous data is available, an optimal threshold on the normality score can be determined
    by maximizing the F-beta score between the actual and predicted anomaly labels.

******************************************
Data
******************************************
.. raw:: html

    <p>
    We use dataset number 179 from the <a href="https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"
    target="_blank">Hexagon ML / UCR Time Series Anomaly Detection Archive</a>.
    The dataset includes a single time series representing a human subject ECG trace sourced from
    record <i>s30791</i> in the <a href="https://physionet.org/content/ltstdb/1.0.0/" target="_blank">
    Long Term ST Database (LTST DB)</a>. The length of the time series is 55,000
    observations. The first 23,000 observations are included in the training set, while the remaining
    32,000 observations are included in the test set. The training set contains only normal data,
    while the test set contains an anomalous heartbeat between observations 52,600 and 52,800.
    </p>

.. raw:: html

    <img
        id="lstm-ad-ecg-anomaly-detection-dataset"
        class="blog-post-image"
        alt="Hexagon ML / UCR dataset N°179 (combined training and test sets)"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/ecg-anomaly-detection/data_light.png
    />

    <p class="blog-post-image-caption">Hexagon ML / UCR dataset N°179 (combined training and test sets).</p>


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
        id="lstm-ad-ecg-anomaly-detection-results"
        class="blog-post-image"
        alt="Results on Hexagon ML / UCR dataset №179 (test set)"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/ecg-anomaly-detection/results_light.png
    />

    <p class="blog-post-image-caption">Results on Hexagon ML / UCR dataset №179 (test set).</p>


******************************************
References
******************************************

[1] Li, H., & Boulanger, P. (2020).
A survey of heart anomaly detection using ambulatory electrocardiogram (ECG).
*Sensors 2020*, 20, 1461.
`doi: 10.3390/s20051461 <https://doi.org/10.3390/s20051461>`__.

[2] Nezamabadi, K., Sardaripour, N., Haghi, B., & Forouzanfar, M. (2022).
Unsupervised ECG analysis: A review.
*IEEE Reviews in Biomedical Engineering*, vol. 16, pp. 208-224.
`doi: 10.1109/RBME.2022.3154893 <https://doi.org/10.1109/RBME.2022.3154893.>`__.

[3] Malhotra, P., Vig, L., Shroff, G., & Agarwal, P. (2015).
Long Short Term Memory Networks for Anomaly Detection in Time Series.
In *European Symposium on Artificial Neural Networks, Computational Intelligence
and Machine Learning. Bruges (Belgium), 22-24 April 2015*, pp. 89-94.
