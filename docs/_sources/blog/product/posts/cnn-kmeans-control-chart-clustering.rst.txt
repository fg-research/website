.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Control chart pattern clustering with the CNN-KMeans SageMaker Algorithm
   :keywords: Amazon SageMaker, Time Series, Clustering, Convolutional Neural Network, K-Means

######################################################################################
Control chart pattern clustering with the CNN-KMeans SageMaker Algorithm
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
    series is referred to as time series representation learning.
    </p>

    <p>
    Several unsupervised and self-supervised deep learning architectures have been proposed in the literature on
    time series representation learning <a href="#references">[4]</a>. One of the most general frameworks is
    introduced in <a href="#references">[5]</a>, where a convolutional encoder is trained in an unsupervised
    manner by minimizing the triplet loss in order to extract a fixed-length feature vector from a set of
    possibly varying-length time series. The feature vectors produced by the convolutional encoder can then be used
    in both unsupervised and supervised downstream tasks, such as time series clustering, time series classification
    and time series regression.
    </p>

    <p>
    In the rest of this post, we will demonstrate how to use the framework introduced in <a href="#references">[5]</a>
    for control chart pattern recognition. We will use our Amazon SageMaker implementation of the clustering version
    of this framework, the <a href="https://fg-research.com/algorithms/time-series-clustering/index.html#cnn-kmeans-sagemaker-algorithm"
    target="_blank">CNN-KMeans SageMaker algorithm</a>, for clustering the control charts in the
    <a href="http://www.timeseriesclassification.com/description.php?Dataset=SyntheticControl" target="_blank">
    Synthetic Control dataset</a> <a href="#references">[6]</a>.
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

    <p>
    The encoder parameters are learned in an unsupervised manner by minimizing the triplet loss. The contrastive learning procedure
    makes the extracted features of a given sequence (anchor or reference) as close as possible to the extracted features of this
    same sequence's subsequences (positive samples) and as distant as possible from the extracted features of other sequences
    (negative samples). All (sub)sequences are sampled randomly during each training iteration.
    </p>

    <p>
    The number of features extracted by the encoder is determined by the number of hidden units of the linear output layer.
    These extracted features are then used for training the K-Means clusterer.
    </p>

******************************************
Data
******************************************

.. raw:: html

    <p>
    We use the <a href="http://www.timeseriesclassification.com/description.php?Dataset=SyntheticControl" target="_blank">
    Synthetic Control dataset</a> introduced in <a href="#references">[2]</a> and available in the <a href="http://www.timeseriesclassification.com/dataset.php"
    target="_blank"> UCR Time Series Classification Archive <a href="#references">[6]</a>.
    The dataset contains 600 synthetically generated time series representing 6 different control chart patterns:
    normal (class 1), cyclic (class 2), increasing trend (class 3), decreasing trend (class 4), upward shift (class 5)
    and downward shift (class 6). The time series are equally split into a training set and a test set.
    </p>

    <img
        id="cnn-kmeans-control-chart-clustering-dataset"
        class="blog-post-image"
        style="width:100%"
        alt="Synthetic Control dataset (combined training and test sets)"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/control-chart-clustering/data_light.png
    />

   <p class="blog-post-image-caption"> Synthetic Control dataset (combined training and test sets).</p>

******************************************
Code
******************************************

==========================================
Environment Set-Up
==========================================

We start by importing all the requirements and setting up the SageMaker environment.

.. warning::

    To be able to run the code below, you need to have an active subscription to the CNN-KMeans SageMaker algorithm.
    You can subscribe to a free trial from the `AWS Marketplace <https://aws.amazon.com/marketplace/pp/prodview-3hdblqdz5nx4m>`__
    in order to get your Amazon Resource Name (ARN). In this post we use version 1.6 of the CNN-KMeans SageMaker algorithm,
    which runs in the PyTorch 2.1.0 Python 3.10 deep learning container.

.. code:: python

    import io
    import sagemaker
    import pandas as pd
    import numpy as np
    from sklearn.metrics import silhouette_score

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

After that we load the training and test datasets, drop the first column with the class labels, and save them in the S3 bucket in CSV format.

.. warning::

    To be able to run the code below, you need to download the datasets (`"SyntheticControl_TRAIN.txt"` and `"SyntheticControl_TEST.txt"`)
    from the `UCR Time Series Classification Archive <http://www.timeseriesclassification.com/description.php?Dataset=SyntheticControl>`__
    and store them in the SageMaker notebook instance.

.. code:: python

    # load the training dataset
    training_dataset = pd.DataFrame(
        data=np.genfromtxt("SyntheticControl_TRAIN.txt")
    )

    # load the test dataset
    test_dataset = pd.DataFrame(
        data=np.genfromtxt("SyntheticControl_TEST.txt")
    )

    # save the training dataset in S3
    training_data = sagemaker_session.upload_string_as_file_body(
        body=training_dataset.iloc[:, 1:].to_csv(index=False, header=False),
        bucket=bucket,
        key="SyntheticControl_train.csv"
    )

    # save the test dataset in S3
    test_data = sagemaker_session.upload_string_as_file_body(
        body=test_dataset.iloc[:, 1:].to_csv(index=False, header=False),
        bucket=bucket,
        key="SyntheticControl_test.csv"
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
            "clusters": 6,
            "algorithm": "lloyd",
            "blocks": 10,
            "filters": 40,
            "kernel-size": 3,
            "reduced-size": 160,
            "output-size": 320,
            "negative-samples": 10,
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

The results are saved in an output file in S3 with the same name as the input file and with the `".out"` file extension.
The results include the predicted cluster labels, which are stored in the first column, and the extracted features,
which are stored in the subsequent columns.

.. code:: python

    # load the model outputs from S3
    predictions = sagemaker_session.read_s3_file(
        bucket=bucket,
        key_prefix=f"{transformer.latest_transform_job.name}/SyntheticControl_test.csv.out"
    )

    # convert the model outputs to data frame
    predictions = pd.read_csv(io.StringIO(predictions), header=None, dtype=float)

After loading the model outputs from S3, we can calculate the clustering metrics.

.. code:: python

    # calculate the Silhouette coefficient
    score = silhouette_score(
        labels=predictions.iloc[:, 0],
        X=predictions.iloc[:, 1:]
    )

We find that the model achieves a Silhouette coefficient of 0.33 on the test set.

.. raw:: html

   <img
        id="cnn-kmeans-control-chart-clustering-results"
        class="blog-post-image"
        style="width:100%"
        alt="Results on Synthetic Control UCR dataset (test set)"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/control-chart-clustering/results_light.png
   />

   <p class="blog-post-image-caption"> Results on Synthetic Control UCR dataset (test set).</p>

After the analysis has been completed, we can delete the model.

.. code:: python

    # delete the model
    transformer.delete_model()

.. tip::

    You can download the
    `notebook <https://github.com/fg-research/cnn-kmeans-sagemaker/blob/master/examples/SyntheticControl.ipynb>`__
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
