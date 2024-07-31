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
    a large amount of data. Machine learning algorithms can automate the process of
    ECG analysis, reducing the need for manual inspection by healthcare professionals.
    </p>

    <p>
    Different supervised and unsupervised machine learning algorithms have been studied
    in the literature on ECG anomaly detection <a href="#references">[1]</a>, <a href="#references">[2]</a>.
    In this post, we will focus on the
    <a href="https://www.esann.org/sites/default/files/proceedings/legacy/es2015-56.pdf" target="_blank">
    Long Short-Term Memory Network for Anomaly Detection (LSTM-AD)</a> <a href="#references">[3]</a>,
    a standard deep learning framework for detecting anomalies in time series data.
    We will demonstrate how to use our Amazon SageMaker implementation of the LSTM-AD model, the
    <a href="https://fg-research.com/algorithms/time-series-anomaly-detection/index.html#lstm-ad-sagemaker-algorithm"
    target="_blank">LSTM-AD SageMaker algorithm</a>,
    for detecting anomalies in an ECG trace.
    </p>

******************************************
Model
******************************************
.. raw:: html

    <p>
    The LSTM-AD model predicts the future values of the time series with a stacked LSTM model.
    The model parameters are learned on a training set containing only normal data (i.e. without anomalies)
    by minimizing the Mean Squared Error (MSE) between the actual and predicted values of the time series.
    After the model has been trained, a Gaussian distribution is fitted to the model’s prediction errors
    on an independent validation set (also without anomalies) using Maximum Likelihood Estimation (MLE).
    </p>

    <img
        id="lstm-ad-ecg-anomaly-detection-diagram"
        class="blog-post-image"
        alt="LSTM-AD architecture."
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/ecg-anomaly-detection/diagram_light.png
    />

    <p class="LSTM-AD architecture.</p>

    <p>
    At inference time, the model predicts the values of the time series (which can now include anomalies)
    at each time step, and calculates the likelihood of the model’s prediction errors under the fitted
    Gaussian distribution.
    The computed Gaussian likelihood is then used as a normality score: the lower the Gaussian
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

We start by importing all the requirements and setting up the SageMaker environment.

.. warning::

    To be able to run the code below, you need to have an active subscription to the LSTM-AD SageMaker algorithm.
    You can subscribe to a free trial from the `AWS Marketplace <https://aws.amazon.com/marketplace/pp/prodview-4pbvedtnnlphw>`__
    in order to get your Amazon Resource Name (ARN). In this post we use version 1.11 of the LSTM-FCN SageMaker algorithm,
    which runs in the PyTorch 2.1.0 Python 3.10 deep learning container.

.. code::

    import io
    import sagemaker
    import pandas as pd
    import numpy as np

    # SageMaker algorithm ARN from AWS Marketplace
    algo_arn = "arn:aws:sagemaker:<...>"

    # SageMaker session
    sagemaker_session = sagemaker.Session()

    # SageMaker role
    role = sagemaker.get_execution_role()

    # S3 bucket
    bucket = sagemaker_session.default_bucket()

    # EC2 instance
    instance_type = "ml.m5.2xlarge"

==========================================
Data Preparation
==========================================

After that we load the dataset and split it into training and test datasets, which we save to S3.

.. warning::

    To be able to run the code below, you need to download the dataset (`"179_UCR_Anomaly_ltstdbs30791AS_23000_52600_52800.txt"`)
    from the `Hexagon ML / UCR Time Series Anomaly Detection Archive <https://www.cs.ucr.edu/~eamonn/time_series_data_2018/>`__
    and store it in the SageMaker notebook instance.

.. code::

    dataset_name = "179_UCR_Anomaly_ltstdbs30791AS_23000_52600_52800"
    cutoff = 23000  # train-test cutoff
    start = 52600   # start of anomalous time interval
    end = 52800     # end of anomalous time interval

    # load the dataset
    dataset = pd.DataFrame(data=np.genfromtxt(f"{dataset_name}.txt"))

    # extract the training dataset
    training_dataset = dataset.iloc[:cutoff]

    # extract the test dataset
    test_dataset = dataset.iloc[cutoff:]

    # save the training dataset in S3
    training_data = sagemaker_session.upload_string_as_file_body(
        body=training_dataset.to_csv(index=False, header=False),
        bucket=bucket,
        key=f"{dataset_name}_train.csv"
    )

    # save the test dataset in S3
    test_data = sagemaker_session.upload_string_as_file_body(
        body=test_dataset.to_csv(index=False, header=False),
        bucket=bucket,
        key=f"{dataset_name}_test.csv"
    )

==========================================
Training
==========================================

Now that the training dataset is available in an accessible S3 bucket, we are ready to fit the model.

.. note::

   The algorithm uses the first 80% of the training dataset for learning
   the LSTM parameters, and the remaining 20% of the training dataset
   for estimating the Gaussian distribution parameters.

.. code::

    # create the estimator
    estimator = sagemaker.algorithm.AlgorithmEstimator(
        algorithm_arn=algo_arn,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        input_mode="File",
        sagemaker_session=sagemaker_session,
        hyperparameters={
            "context-length": 100,
            "prediction-length": 10,
            "sequence-stride": 10,
            "hidden-size": 32,
            "num-layers": 2,
            "dropout": 0.5,
            "lr": 0.001,
            "batch-size": 128,
            "epochs": 100,
        },
    )

    # run the training job
    estimator.fit({"training": training_data})

==========================================
Inference
==========================================

Once the training job has completed, we can run a batch transform job on the test dataset.

.. code::

    # create the transformer
    transformer = estimator.transformer(
        instance_count=1,
        instance_type=instance_type,
        max_payload=100,
    )

    # run the transform job
    transformer.transform(
        data=test_data,
        content_type="text/csv",
    )

The results are saved in an output file in S3 with the same name
as the input file and with the `".out"` file extension.
The output file contains the normality scores in the first
column, and the predicted values of the time series in the
subsequent columns.

.. code::

    # load the model outputs from S3
    predictions = sagemaker_session.read_s3_file(
        bucket=bucket,
        key_prefix=f"{transformer.latest_transform_job.name}/{dataset_name}_test.csv.out"
    )

    # convert the model outputs to data frame
    predictions = pd.read_csv(io.StringIO(predictions), header=None, dtype=float)

After loading the normality scores and the predicted values from S3, we can visualize the results.

.. note::

    The algorithm defines the normality scores using the Gaussian log-likelihood instead of the likelihood.

.. raw:: html

    <img
        id="lstm-ad-ecg-anomaly-detection-results"
        class="blog-post-image"
        alt="Results on Hexagon ML / UCR dataset №179 (test set)"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/ecg-anomaly-detection/results_light.png
    />

    <p class="blog-post-image-caption">Results on Hexagon ML / UCR dataset №179 (test set).</p>

We find that the model correctly identifies the anomalies, as the normality score exhibits the largest
downward spikes on the same time steps where the anomalies are observed.

After the analysis has been completed, we can delete the model.

.. code:: python

    # delete the model
    transformer.delete_model()

.. tip::

    You can download the
    `notebook <https://github.com/fg-research/lstm-ad-sagemaker/blob/master/examples/179_UCR_Anomaly_ltstdbs30791AS_23000_52600_52800.ipynb>`__
    with the full code from our
    `GitHub <https://github.com/fg-research/lstm-ad-sagemaker>`__
    repository.

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
