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
    Different neural network architectures have been proposed in the literature
    on ECG arrhythmia classification <a href="#references">[1]</a>. In this post,
    we will focus on the <a href="https://doi.org/10.1109/ACCESS.2017.2779939"
    target="_blank"> Long Short-Term Memory Fully Convolutional Network</a>
    <a href="#references">[2]</a>, which we will refer to as the LSTM-FCN model.
    We will demonstrate how to use our Amazon SageMaker implementation of the LSTM-FCN model,
    the <a href="https://fg-research.com/algorithms/time-series-classification/index.html#lstm-fcn-sagemaker-algorithm"
    target="_blank">LSTM-FCN SageMaker algorithm</a>, for categorizing the ECG traces in the
    <a href="https://physionet.org/content/mitdb/1.0.0" target="_blank">PhysioNet MIT-BIH Arrhythmia Database</a>
    <a href="#references">[3]</a>.

******************************************
Model
******************************************

.. raw:: html

    <p>
    The LSTM-FCN model includes two blocks: a recurrent block and a convolutional block.
    The recurrent block consists of a single LSTM layer followed by a dropout layer.
    The convolutional block consists of three convolutional layers, each followed by
    batch normalization and ReLU activation, and of a global average pooling layer.
    </p>

    <p>
    The input time series are passed to both blocks. The convolutional block processes each
    time series as a single feature observed across multiple time steps, while the recurrent
    block processes each time series as multiple features observed at a single time step
    (referred to as <i>dimension shuffling</i>). The outputs of the two blocks are
    concatenated and passed to a final output layer with softmax activation.
    </p>

******************************************
Data
******************************************
.. raw:: html

    <p>
    We use the <a href="https://www.kaggle.com/datasets/shayanfazeli/heartbeat" target="_blank">
    pre-processed version of the PhysioNet MIT-BIH Arrhythmia Database</a> made available in <a href="#references">[4]</a>
    where the ECG recordings are split into individual heartbeats and then downsampled and padded
    with zeroes to the fixed length of 187. The dataset contains 5 different categories of heartbeats
    where class 0 indicates a normal heartbeat while classes 1, 2, 3, and 4 correspond to different
    types of arrhythmia.
    </p>

    <p>
    The dataset is split into a training set and a test set. The train-test split is provided
    directly by the authors. The training set contains 87,554 time series while the test set
    contains 21,892 time series. Both the training and test sets are imbalanced, as most time
    series represent normal heartbeats.
    </p>

   <img id="lstm-fcn-ecg-classification-class-distribution" class="blog-post-image" alt="MIT-BIH Dataset Class Distribution" src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/ecg-classification/mit_bih_dataset_light.png />

   <p class="blog-post-image-caption"> MIT-BIH dataset class distribution.</p>

******************************************
Code
******************************************

==========================================
Environment Set-Up
==========================================

We start by installing and importing all the requirements and setting up the SageMaker environment.

.. warning::

    To be able to run the code below, you need to have an active subscription to the LSTM-FCN SageMaker algorithm.
    You can subscribe to a free trial from the `AWS Marketplace <https://aws.amazon.com/marketplace/pp/prodview-vzxmyw25oqtx6>`__
    in order to get your Amazon Resource Name (ARN). In this post we use version 1.14 of the LSTM-FCN SageMaker algorithm,
    which runs in the PyTorch 2.1.0 Python 3.10 deep learning container.

.. code::

    pip install imbalanced-learn

.. code:: python

    import io
    import sagemaker
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.metrics import accuracy_score, confusion_matrix

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
After that we load the training data from the CSV file.

.. warning::
    To be able to run the code below, you need to download the datasets (`"mitbih_train.csv"` and `"mitbih_test.csv"`)
    from `Kaggle <https://www.kaggle.com/datasets/shayanfazeli/heartbeat>`__ and store them in the SageMaker notebook
    instance.

.. code:: python

    # load the training data
    training_dataset = pd.read_csv("mitbih_train.csv", header=None)

To speed up the training process, we undersample the training data using `imbalanced-learn <https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html>`__.
After resampling, the training data contains 641 instances of each class.

.. code:: python

    # resample the training data
    sampler = RandomUnderSampler(random_state=42)
    training_dataset = pd.concat(sampler.fit_resample(X=training_dataset.iloc[:, :-1], y=training_dataset.iloc[:, -1:]), axis=1)

We then proceed to moving the class labels from the last column to the first column, as required by the LSTM-FCN SageMaker algorithm.

.. code:: python

    # move the class labels to the first column
    training_dataset = pd.concat([training_dataset.iloc[:, -1:], training_dataset.iloc[:, 1:]], axis=1)

Once this is done, we can save the training data in the S3 bucket in CSV format.

.. code:: python

    # save the training data in S3
    training_data = sagemaker_session.upload_string_as_file_body(
        body=training_dataset.to_csv(index=False, header=False),
        bucket=bucket,
        key="MITBIH_train.csv"
    )

==========================================
Training
==========================================
We can now run the training job.

.. code:: python

    # create the estimator
    estimator = sagemaker.algorithm.AlgorithmEstimator(
        algorithm_arn=algo_arn,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        input_mode="File",
        sagemaker_session=sagemaker_session,
        hyperparameters={
            "num-layers": 1,
            "hidden-size": 128,
            "dropout": 0.8,
            "filters-1": 128,
            "filters-2": 256,
            "filters-3": 128,
            "kernel-size-1": 8,
            "kernel-size-2": 5,
            "kernel-size-3": 3,
            "batch-size": 256,
            "lr": 0.001,
            "epochs": 100,
        },
    )

    # run the training job
    estimator.fit({"training": training_data})

==========================================
Inference
==========================================
Once the training job has completed, we can deploy the model to a real-time endpoint.

.. code:: python

    # define the endpoint inputs serializer
    serializer = sagemaker.serializers.CSVSerializer(content_type="text/csv")

    # define the endpoint outputs deserializer
    deserializer = sagemaker.deserializers.CSVDeserializer(accept="text/csv")

    # create the endpoint
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        serializer=serializer,
        deserializer=deserializer,
    )

After that we load the test data from the CSV file.

.. code:: python

    # load the test data
    test_dataset = pd.read_csv("mitbih_test.csv", header=None)

To avoid confusion, we move the class labels from the last column to the first column, even though they are obviously not used for inference.

.. code:: python

    # move the class labels to the first column
    test_dataset = pd.concat([test_dataset.iloc[:, -1:], test_dataset.iloc[:, 1:]], axis=1)

Given that the test dataset is relatively large, we invoke the endpoint with batches of time series as opposed to using the entire test dataset as a single payload.

.. code:: python

    # define the batch size
    batch_size = 100

    # create a data frame for storing the model predictions
    predictions = pd.DataFrame()

    # loop across the test dataset
    for i in range(0, len(test_dataset), batch_size):

        # invoke the endpoint with a batch of time series, make sure to drop the first column with the class labels
        response = sagemaker_session.sagemaker_runtime_client.invoke_endpoint(
            EndpointName=predictor.endpoint_name,
            ContentType="text/csv",
            Body=serializer.serialize(test_dataset.iloc[i:i + batch_size, 1:])
        )

        # save the predicted class labels in the data frame
        predictions = pd.concat([
            predictions,
            pd.DataFrame(
                data=deserializer.deserialize(response["Body"], content_type="text/csv"),
                dtype=float
            )
        ], axis=0)

After generating the model predictions, we can calculate the classification metrics.

.. code:: python

    # calculate the accuracy
    accuracy_score(y_true=test_dataset.iloc[:, 0], y_pred=predictions.iloc[:, 0])

    # calculate the confusion matrix
    confusion_matrix(y_true=test_dataset.iloc[:, 0], y_pred=predictions.iloc[:, 0])

We find that the model achieves 99.79% accuracy on the test data.

.. raw:: html

    <img id="lstm-fcn-ecg-classification-confusion-matrix" class="blog-post-image" alt="LSTM-FCN confusion matrix on MIT-BIH test dataset" src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/ecg-classification/confusion_matrix_light.png />

    <p class="blog-post-image-caption"> LSTM-FCN confusion matrix on MIT-BIH test dataset.</p>

After the analysis has been completed, we can delete the model and the endpoint.

.. code:: python

    # delete the model
    predictor.delete_model()

    # delete the endpoint
    predictor.delete_endpoint()

.. tip::

    You can download the
    `notebook <https://github.com/fg-research/lstm-fcn-sagemaker/blob/master/examples/MIT-BIH.ipynb>`__
    with the full code from our
    `GitHub <https://github.com/fg-research/lstm-fcn-sagemaker>`__
    repository.

******************************************
References
******************************************

[1] Ebrahimi, Z., Loni, M., Daneshtalab, M., & Gharehbaghi, A. (2020).
A review on deep learning methods for ECG arrhythmia classification.
*Expert Systems with Applications: X*, vol. 7, 100033.
`doi: 10.1016/j.eswax.2020.100033 <https://doi.org/10.1016/j.eswax.2020.100033>`__.

[2] Karim, F., Majumdar, S., Darabi, H., & Chen, S. (2018).
LSTM fully convolutional networks for time series classification.
*IEEE Access*, vol. 6, pp. 1662-1669,
`doi: 10.1109/ACCESS.2017.2779939 <https://doi.org/10.1109/ACCESS.2017.2779939>`__.

[3] Moody, G. B., & Mark, R. G. (2001).
The impact of the MIT-BIH arrhythmia database.
*IEEE engineering in medicine and biology magazine*, vol. 20, no. 3, pp. 45-50,
`doi: 10.1109/51.932724 <https://doi.org/10.1109/51.932724>`__.

[4] Kachuee, M., Fazeli, S., & Sarrafzadeh, M. (2018).
ECG heartbeat classification: A deep transferable representation.
*2018 IEEE international conference on healthcare informatics (ICHI)*, pp. 443-444,
`doi: 10.1109/ICHI.2018.00092 <https://doi.org/10.1109/ICHI.2018.00092>`__.
