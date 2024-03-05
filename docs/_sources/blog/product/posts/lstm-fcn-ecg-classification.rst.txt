.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Arrhythmia classification with the LSTM-FCN SageMaker Algorithm
   :keywords: Amazon SageMaker, Time Series, Classification, LSTM

######################################################################################
Arrhythmia classification with the LSTM-FCN SageMaker Algorithm
######################################################################################

.. raw:: html

    <p>
    Arrhythmia classification based on electrocardiogram (ECG) data involves identifying and
    categorizing abnormal patterns of cardiac electrical activity detected in the ECG signal.
    Arrhythmia classification is important for diagnosing cardiac abnormalities, assessing the
    risk of adverse cardiovascular events and guiding appropriate treatment strategies.
    Machine learning algorithms can automate the process of ECG interpretation, reducing the
    reliance on manual analysis by healthcare professionals, a task that is both time-consuming
    and prone to errors. The automation provided by machine learning algorithms offers the potential
    fast, accurate and cost-effective diagnosis.
    <p>

    <p>
    In this post, we use the Long Short-Term Memory Fully Convolutional Network (LSTM-FCN)
    <a href="#references">[1]</a> for classifying the ECG traces in the
    `PhysioNet MIT-BIH Arrhythmia Database <https://physionet.org/content/mitdb/1.0.0>`__
    <a href="#references">[2]</a>. We use the `pre-processed version of the dataset <https://www.kaggle.com/datasets/shayanfazeli/heartbeat>`__
    made available in [3] where the ECG recordings are split into individual heartbeats and
    then downsampled and padded with zeroes to the fixed length of 187. The dataset contains
    5 different categories of heartbeats where class 0 indicates a normal heartbeat while
    classes 1, 2, 3, and 4 correspond to different types of arrhythmia.
    <p>






******************************************
References
******************************************

[1] Karim, F., Majumdar, S., Darabi, H., & Chen, S. (2018). LSTM fully convolutional networks for time series classification.
*IEEE Access*, vol. 6, pp. 1662-1669,
`doi: 10.1109/ACCESS.2017.2779939 <https://doi.org/10.1109/ACCESS.2017.2779939>`__.

[2] Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH arrhythmia database.
*IEEE engineering in medicine and biology magazine*, vol. 20, no. 3, pp. 45-50,
`doi: 10.1109/51.932724 <https://doi.org/10.1109/51.932724>`__.

[3] Kachuee, M., Fazeli, S., & Sarrafzadeh, M. (2018). ECG heartbeat classification: A deep transferable representation.
*2018 IEEE international conference on healthcare informatics (ICHI)*, pp. 443-444,
`doi: 10.1109/ICHI.2018.00092 <https://doi.org/10.1109/ICHI.2018.00092>`__.









