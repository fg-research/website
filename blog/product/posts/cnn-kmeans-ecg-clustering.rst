.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Clustering ECG signals with the CNN-KMeans SageMaker Algorithm
   :keywords: Amazon SageMaker, Time Series, Clustering, Convolutional Neural Network, K-Means

######################################################################################
Clustering ECG signals with the CNN-KMeans SageMaker Algorithm
######################################################################################

.. raw:: html

    <p>
    Time series clustering is the task of partitioning a set of time series into homogeneous groups.
    Traditional clustering algorithms based on the Euclidean distance, such as K-Means clustering, are
    not directly applicable to time series data, as time series with similar patterns can have large
    Euclidean distance due to misalignments and offsets along the time axis <a href="#references">[1]</a>,
    <a href="#references">[2]</a>, <a href="#references">[3]</a>.
    </p>

    <p>
    A possible solution to this problem is to encode the time series into a number of time-independent features,
    and to use these derived features as inputs in a standard clustering algorithm based on the Euclidean distance
    <a href="#references">[3]</a>. The task of extracting the time-independent features of a set of unlabelled time
    series is referred to as <i>time series representation learning</i>.
    </p>

    <p>
    Several unsupervised and self-supervised deep learning architectures have been proposed in the literature on
    time series representation learning <a href="#references">[4]</a>. One of the most general frameworks is
    introduced in <a href="#references">[5]</a>, where an unsupervised convolutional encoder is used to
    transform each time series into a fixed-length feature vector. The feature vectors produced by the convolutional
    encoder can then be used in both unsupervised and supervised downstream tasks, such as time series clustering,
    time series classification and time series regression.
    </p>

    <p>
    In the rest of this post, we will demonstrate how to use the framework introduced in <a href="#references">[5]</a>
    for clustering ECG signals. We will use our Amazon SageMaker implementation of the clustering variant of this framework, the
    <a href="https://fg-research.com/algorithms/time-series-clustering/index.html#cnn-kmeans-sagemaker-algorithm"
    target="_blank">CNN-KMeans SageMaker algorithm</a>, for clustering the ECG traces in the
    <a href="http://www.timeseriesclassification.com/description.php?Dataset=ECG200" target="_blank">
    ECG200 dataset</a> <a href="#references">[6]</a>.
    </p>

******************************************
Model
******************************************

.. raw:: html

    <p>
    The model has two components: an encoder which extracts the relevant features, and a K-Means clusterer which takes as input
    the extracted features and predicts the cluster labels.
    </p>

    <p>
    The encoder consists of a stack of exponentially dilated causal convolutional blocks, followed by an adaptive max pooling layer
    and a linear output layer. Each block includes two causal convolutional layers with the same dilation rate, each followed
    by weight normalization and Leaky ReLU activation. A residual connection is applied between the input and the output of each
    block, where the input is transformed by an additional convolutional layer with a kernel size of 1 when its length does not
    match the one of the output.
    </p>

    <img
        id="cnn-kmeans-ecg-clustering-diagram"
        class="blog-post-image"
        alt="Encoder block."
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/cnn-kmeans-ecg-clustering/diagram.png
    />

    <p class="blog-post-image-caption">Encoder block.</p>

    <p>
    The encoder parameters are learned in an unsupervised manner by minimizing the <i>triplet loss</i>. The contrastive learning
    procedure makes the extracted features of a given sequence (<i>anchor</i> or <i>reference</i>) as close as possible to the
    extracted features of this same sequence's subsequences (<i>positive samples</i>) and as distant as possible from the extracted
    features of other sequences (<i>negative samples</i>). All (sub)sequences are sampled randomly during each training iteration.
    </p>

    <p>
    The number of features extracted by the encoder is determined by the number of hidden units of the linear output layer.
    These extracted features are then used as input by the K-Means clusterer.
    </p>

******************************************
Data
******************************************

.. raw:: html

    <p>
    We use the "ECG200" dataset introduced in <a href="#references">[7]</a> and available
    in the UCR Time Series Classification Archive <a href="#references">[6]</a>.
    The dataset contains 200 time series of cardiac electrical activity as recorded from electrodes
    at various locations on the body. Each time series in the dataset contains 96 measurements
    recorded by one electrode during one heartbeat. 133 time series are labelled as normal (class 1),
    while 67 time series are labelled as abnormal (class -1). The time series are equally split into
    a training set and a test set.
    </p>

    <img
        id="cnn-kmeans-ecg-clustering-time-series"
        class="blog-post-image"
        alt="ECG200 dataset (combined training and test sets)"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/cnn-kmeans-ecg-clustering/data_light.png
    />

   <p class="blog-post-image-caption"> ECG200 dataset (combined training and test sets).</p>

******************************************
Code
******************************************

.. warning::

    To be able to run the code below, you need to have an active subscription to the CNN-KMeans SageMaker algorithm.
    You can subscribe to a free trial from the `AWS Marketplace <https://aws.amazon.com/marketplace/pp/prodview-3hdblqdz5nx4m>`__
    in order to get your Amazon Resource Name (ARN). In this post we use version 1.6 of the CNN-KMeans SageMaker algorithm,
    which runs in the PyTorch 2.1.0 Python 3.10 deep learning container.

    You will also need to download the "ECG200" dataset
    from the `UCR Time Series Classification Archive <http://www.timeseriesclassification.com/description.php?Dataset=ECG200>`__
    and store the files in the SageMaker notebook instance.

==========================================
Environment Set-Up
==========================================

We start by importing all the requirements and setting up the SageMaker environment.

.. code:: python

    import io
    import sagemaker
    import pandas as pd
    import numpy as np

    # SageMaker algorithm ARN, replace the placeholder below with your AWS Marketplace ARN
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

After that we load the training and test datasets, drop the first column with the class labels, and save them in the S3 bucket in CSV format.

.. code:: python

    # load the training dataset
    training_dataset = pd.DataFrame(
        data=np.genfromtxt("ECG200_TRAIN.txt")
    )

    # load the test dataset
    test_dataset = pd.DataFrame(
        data=np.genfromtxt("ECG200_TEST.txt")
    )

    # save the training dataset in S3
    training_data = sagemaker_session.upload_string_as_file_body(
        body=training_dataset.iloc[:, 1:].to_csv(index=False, header=False),
        bucket=bucket,
        key="ECG200_train.csv"
    )

    # save the test dataset in S3
    test_data = sagemaker_session.upload_string_as_file_body(
        body=test_dataset.iloc[:, 1:].to_csv(index=False, header=False),
        bucket=bucket,
        key="ECG200_test.csv"
    )

==========================================
Training
==========================================

Now that the training dataset is available in an accessible S3 bucket, we are ready to fit the model.

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
            "clusters": 2,
            "algorithm": "lloyd",
            "blocks": 1,
            "filters": 50,
            "kernel-size": 3,
            "reduced-size": 100,
            "output-size": 50,
            "negative-samples": 10,
            "lr": 0.001,
            "batch-size": 64,
            "epochs": 50,
        },
    )

    # run the training job
    estimator.fit({"training": training_data})

==========================================
Inference
==========================================

Once the training job has completed, we can run a batch transform job on the test dataset.

.. code:: python

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

The results are saved in an output file in S3 with the same name as the input file and with the :code:`".out"` file extension.
The results include the predicted cluster labels, which are stored in the first column, and the extracted features,
which are stored in the subsequent columns.

.. code:: python

    # load the model outputs from S3
    predictions = sagemaker_session.read_s3_file(
        bucket=bucket,
        key_prefix=f"{transformer.latest_transform_job.name}/ECG200_test.csv.out"
    )

    # convert the model outputs to data frame
    predictions = pd.read_csv(io.StringIO(predictions), header=None, dtype=float)

==========================================
Evaluation
==========================================

After loading the model outputs from S3, we can compare the predicted cluster labels to the ground truth class labels.

.. code:: python

    results = pd.crosstab(
        index=pd.Series(data=test_dataset.iloc[:, 0].values, name="class label"),
        columns=pd.Series(data=predictions.iloc[:, 0].values, name="cluster label"),
        normalize="index"
    )

We find that the model achieves approximately 83% accuracy, as it assigns 82.81% of
the normal ECG traces (class 1) to cluster 0, and 83.33% of the abnormal ECG traces
(class -1) to cluster 1.

.. raw:: html

   <img
        id="cnn-kmeans-ecg-clustering-results"
        class="blog-post-image"
        alt="Results on ECG200 dataset (test set)"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/cnn-kmeans-ecg-clustering/results_light.png
   />

   <p class="blog-post-image-caption"> Results on ECG200 dataset (test set).</p>

After the analysis has been completed, we can delete the model.

.. code:: python

    # delete the model
    transformer.delete_model()

.. tip::

    You can download the
    `notebook <https://github.com/fg-research/cnn-kmeans-sagemaker/blob/master/examples/ECG200.ipynb>`__
    with the full code from our
    `GitHub <https://github.com/fg-research/cnn-kmeans-sagemaker>`__
    repository.

******************************************
References
******************************************

[1] Kontaki, M., Papadopoulos, A. N., & Manolopoulos, Y. (2005).
Similarity search in time series databases.
In *Encyclopedia of Database Technologies and Applications*, pp. 646-651.
`doi: 10.4018/978-1-59140-560-3.ch106 <https://doi.org/10.4018/978-1-59140-560-3.ch106>`__.

[2] Alcock, R. J., & Manolopoulos, Y. (1999).
Time-series similarity queries employing a feature-based approach.
In *7th Hellenic conference on informatics*, pp. 27-29.

[3] Lafabregue, B., Weber, J., Gançarski, P., & Forestier, G. (2022).
End-to-end deep representation learning for time series clustering: a comparative study.
*Data mining and knowledge discovery*, vol. 36, pp. 29-81.
`doi: 10.1007/s10618-021-00796-y <https://doi.org/10.1007/s10618-021-00796-y>`__.

[4] Meng, Q., Qian, H., Liu, Y., Xu, Y., Shen, Z., & Cui, L. (2023).
Unsupervised representation learning for time series: A review.
*arXiv preprint*.
`doi: 10.48550/arXiv.2308.01578 <https://doi.org/10.48550/arXiv.2308.01578>`__.

[5] Franceschi, J. Y., Dieuleveut, A., & Jaggi, M. (2019).
Unsupervised scalable representation learning for multivariate time series.
*Advances in neural information processing systems*, vol. 32.

[6] Dau, H. A., Bagnall, A., Kamgar, K., Yeh, C. C. M., Zhu, Y., Gharghabi, S., Ratanamahatana, C. A., & Keogh, E. (2019).
The UCR time series archive.
*IEEE/CAA Journal of Automatica Sinica*, vol. 6, no. 6, pp. 1293-1305.
`doi: 10.1109/JAS.2019.1911747 <https://doi.org/10.1109/JAS.2019.1911747>`__.

[7] Olszewski, R. T. (2001). Generalized feature extraction for structural pattern recognition in time-series data.
*Carnegie Mellon University*.
