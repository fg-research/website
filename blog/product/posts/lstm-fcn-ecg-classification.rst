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
    </p>

    <p>
    Machine learning algorithms can automate the process of ECG interpretation, reducing the
    reliance on manual analysis by healthcare professionals, a task that is both time-consuming
    and prone to errors. The automation provided by machine learning algorithms offers the
    potential for fast, accurate and cost-effective diagnosis.
    </p>

    <p>
    In this post, we use our Amazon SageMaker implementation of the Long Short-Term Memory
    Fully Convolutional Network (LSTM-FCN) <a href="#references">[1]</a>, the
    <a href="https://fg-research.com/algorithms/time-series-classification/index.html#lstm-fcn-sagemaker-algorithm" target="_blank">LSTM-FCN SageMaker algorithm</a>,
    for classifying the ECG traces in the <a href="https://physionet.org/content/mitdb/1.0.0" target="_blank">PhysioNet MIT-BIH Arrhythmia Database</a>
    <a href="#references">[2]</a>. We use the <a href="https://www.kaggle.com/datasets/shayanfazeli/heartbeat" target="_blank">
    pre-processed version of the PhysioNet MIT-BIH Arrhythmia Database</a> made available in <a href="#references">[3]</a>
    where the ECG recordings are split into individual heartbeats and then downsampled and padded
    with zeroes to the fixed length of 187. The dataset contains 5 different categories
    of heartbeats where class 0 indicates a normal heartbeat while classes 1, 2, 3, and 4 correspond
    to different types of arrhythmia.
    </p>

******************************************
Model
******************************************

.. raw:: html

    <p>
    The LSTM-FCN model includes two blocks: a recurrent block and a convolutional block.
    The recurrent block consists of a single LSTM layer (either general or with attention)
    followed by a dropout layer. The convolutional block consists of three convolutional
    layers, each followed by batch normalization and ReLU activation, and of a global
    average pooling layer.
    </p>

    <p>
    The input time series are passed to both blocks. The convolutional block processes each
    time series as a single feature observed across multiple time steps, while the recurrent
    block processes each time series as multiple features observed at a single time step
    (referred to as dimension shuffling).
    </p>

    <p>
    The outputs of the two blocks are concatenated and passed to a final output layer with softmax activation.
    The model parameters are learned by minimizing the cross-entropy loss.
    </p>

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









