.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Epileptic seizure detection with the InceptionTime SageMaker Algorithm
   :keywords: Amazon SageMaker, Time Series, Classification, Convolutional Neural Network, Epilepsy

######################################################################################
Epileptic seizure detection with the InceptionTime SageMaker Algorithm
######################################################################################

.. raw:: html

    <p>
    ...
    </p>


******************************************
Model
******************************************
.. raw:: html

    <p>
    InceptionTime <a href="#references">[2]</a> is an ensemble model. The only difference between the models in the ensemble
    is in the initial values of the weights, which are sampled from the Glorot uniform distribution.
    </p>

    <p>
    Each model consists of a stack of Inception blocks.
    Each block includes three convolutional layers with kernel sizes of 10, 20 and 40 and a max pooling layer.
    The block input is processed by the four layers in parallel, and the four outputs are concatenated
    before being passed to a batch normalization layer followed by a ReLU activation.
    </p>

    <img
        id="inception-time-epilepsy-recognition-diagram"
        class="blog-post-image"
        alt="Inception block."
        style="width:100%"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inception-time-epilepsy-recognition/diagram.png
    />

    <p class="blog-post-image-caption">Inception block.</p>

    <p>
    A residual connection is applied between the input time series and the output of the second block,
    and after that between every three blocks.
    The residual connection processes the inputs using an additional convolutional layer with a kernel size of 1
    followed by a batch normalization layer.
    The processed inputs are then added to the output, which is transformed by a ReLU activation.
    The output of the last block is passed to an average pooling layer which removes the time dimension,
    and then to a final linear layer.
    </p>

    <p>
    At inference time, the class probabilities predicted by the different models are averaged in order to obtain
    a unique predicted probability and, therefore, a unique predicted label, for each class.
    </p>

.. note::

    The InceptionTime SageMaker algorithm implements the model as described above with no changes.
    However, the initial values of the weights are not sampled from the Glorot uniform distribution,
    but are determined using PyTorch's default initialization method.

******************************************
Data
******************************************
.. raw:: html

    <p>
    We use the "Epilepsy" dataset introduced in <a href="#references">[3]</a> and available
    in the <a href=http://www.timeseriesclassification.com>UCR Time Series Classification Archive</a>
    <a href="#references">[4]</a>.
    The data was collected from 6 study participants who conducted 4 different activities
    while wearing a tri-axial accelerometer on their wrist: walking, running, sewing and
    mimicking epileptic seizures.
    The mimicked epileptic seizures followed a protocol defined by a medical expert.
    </p>

    <p>
    The dataset contains 275 three-dimensional time series of length 206.
    The data was recorded at a sampling frequency of 16 Hz, and therefore
    the time series span approximately 13 seconds.
    137 time series are included in the training set, while the remaining
    138 time series are included in the test set.
    The training set and test time series correspond to different participants.
    </p>

    <img
        id="inception-time-epilepsy-recognition-time-series"
        class="blog-post-image"
        alt="Epilepsy dataset (combined training and test sets)"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inception-time-epilepsy-recognition/data_light.png
    />

   <p class="blog-post-image-caption"> Epilepsy dataset (combined training and test sets).</p>

******************************************
Code
******************************************

.. warning::

    To be able to run the code below, you need to have an active subscription to the InceptionTime SageMaker algorithm.
    You can subscribe to a free trial from the `AWS Marketplace <https://aws.amazon.com/marketplace/pp/prodview-omz7rumnllmla>`__
    in order to get your Amazon Resource Name (ARN). In this post we use version 1.8 of the InceptionTime SageMaker algorithm,
    which runs in the PyTorch 2.1.0 Python 3.10 deep learning container.

    You will also need to download the "Epilepsy" dataset from the
    `UCR Time Series Classification Archive <http://www.timeseriesclassification.com/description.php?Dataset=Epilepsy>`__
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
    import matplotlib.pyplot as plt
    from scipy.io import arff
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

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
After that we define a function for reading and preparing the data in the format required by the algorithm.
The algorithm expects the column names of the one-hot encoded class labels to start with :code:`"y"`
and the column names of the time series values to start with :code:`"x"`.
The algorithm also requires including unique sample identifiers in a column named :code:`"sample"` and
unique feature identifiers in a column named :code:`"feature"`.

.. code:: python

    def read_data(dimension, split):

        # load the data
        df = pd.DataFrame(data=arff.loadarff(f"EpilepsyDimension{dimension}_{split}.arff")[0])

        # extract the features and labels
        features, labels = df.iloc[:, :-1], df.iloc[:, -1:]

        # rename the features
        features.columns = [f"x_{i}" for i in range(1, 1 + features.shape[1])]

        # one-hot encode the labels
        ohe = OneHotEncoder(sparse_output=False).fit(labels)
        labels = pd.DataFrame(data=ohe.transform(labels), columns=[f'y_{c.decode("utf-8")}' for c in ohe.categories_[0]])

        # merge the labels and features
        data = labels.join(features)

        # add the sample ids
        data.insert(0, "sample", range(1, 1 + len(df)))

        # add the feature ids
        data.insert(1, "feature", dimension)

        return data

---------------------------
Training Data
---------------------------
We now load the training data from the :code:`ARFF` files.

.. code:: python

    # load the training data
    training_dataset = pd.concat([read_data(d, "TRAIN") for d in range(1, 4)]).sort_values(by=["sample", "feature"], ignore_index=True)

.. code:: python

    training_dataset.shape

.. code-block:: console

    (411, 212)

.. code:: python

    training_dataset.head()

.. raw:: html

    <img
        id="inception-time-epilepsy-recognition-training-dataset-head"
        class="blog-post-image"
        alt="First 6 rows of training dataset"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inception-time-epilepsy-recognition/training_dataset_head_light.png
        style="width:100%"
    />

.. code:: python

    training_dataset.tail()

.. raw:: html

    <img
        id="inception-time-epilepsy-recognition-training-dataset-tail"
        class="blog-post-image"
        alt="Last 6 rows of training dataset"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inception-time-epilepsy-recognition/training_dataset_tail_light.png
        style="width:100%"
    />

We save the training dataset to a CSV file in S3, such that it can be used by the training algorithm.

.. code:: python

    # save the training data in S3
    training_data = sagemaker_session.upload_string_as_file_body(
        body=training_dataset.to_csv(index=False),
        bucket=bucket,
        key="Epilepsy_train.csv"
    )

---------------------------
Test Data
---------------------------
We then load the test data from the :code:`ARFF` files.

.. code:: python

    # load the test data
    test_dataset = pd.concat([read_data(d, "TEST") for d in range(1, 4)]).sort_values(by=["sample", "feature"], ignore_index=True)

We split the test data into two different data frames: a data frame containing the time series
that we will use for inference, and a separate data frame containing the class labels
that we will use for validation.

.. code:: python

    # extract the time series
    test_inputs = test_dataset[["sample", "feature"] + [c for c in test_dataset.columns if c.startswith("x")]]

.. code:: python

    test_inputs.head()

.. raw:: html

    <img
        id="inception-time-epilepsy-recognition-test-inputs-head"
        class="blog-post-image"
        alt="First 6 rows of test inputs"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inception-time-epilepsy-recognition/test_inputs_head_light.png
    />

.. code:: python

    test_inputs.tail()

.. raw:: html

    <img
        id="inception-time-epilepsy-recognition-test-inputs-tail"
        class="blog-post-image"
        alt="Last 6 rows of test inputs"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inception-time-epilepsy-recognition/test_inputs_tail_light.png
    />

.. code:: python

    # extract the class labels
    test_outputs = test_dataset[["sample"] + [c for c in test_dataset.columns if c.startswith("y")]].drop_duplicates(ignore_index=True)

.. code:: python

    test_outputs.head()

.. raw:: html

    <img
        id="inception-time-epilepsy-recognition-test-outputs-head"
        class="blog-post-image"
        alt="First 6 rows of test outputs"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inception-time-epilepsy-recognition/test_outputs_head_light.png
    />

.. code:: python

    test_outputs.tail()

.. raw:: html

    <img
        id="inception-time-epilepsy-recognition-test-outputs-tail"
        class="blog-post-image"
        alt="Last 6 rows of test outputs"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inception-time-epilepsy-recognition/test_outputs_tail_light.png
    />

We save the data frame with the time series to a CSV file in S3, such that it can be used by the inference algorithm.

.. code:: python

    # save the test data in S3
    test_data = sagemaker_session.upload_string_as_file_body(
        body=test_inputs.to_csv(index=False),
        bucket=bucket,
        key="Epilepsy_test.csv"
    )

==========================================
Training
==========================================

Now that the training dataset is available in an accessible S3 bucket, we can train the model.
We train an ensemble of 5 models, where each model has 6 blocks. We set the number of filters
of each convolutional layer in each block equal to 32. We run the training for 100 epochs
with a batch size of 256 and a learning rate of 0.001.

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
            "filters": 32,
            "depth": 6,
            "models": 5,
            "batch-size": 256,
            "lr": 0.001,
            "epochs": 100,
            "task": "multiclass"
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

The results are saved in an output file in S3 with the same name as the input file and
with the :code:`".out"` file extension. The results include the predicted class labels, whose
column names start with :code:`"y"`, and the predicted class probabilities, whose column
names start with :code:`"p"`

.. code:: python

    # load the model outputs from S3
    predictions = sagemaker_session.read_s3_file(
        bucket=bucket,
        key_prefix=f"{transformer.latest_transform_job.name}/Epilepsy_test.csv.out"
    )

    # convert the model outputs to data frame
    predictions = pd.read_csv(io.StringIO(predictions))

.. code:: python

    predictions.shape

.. code-block:: console

    (138, 9)

.. code:: python

    predictions.head()

.. raw:: html

    <img
        id="inception-time-epilepsy-recognition-predictions-head"
        class="blog-post-image"
        alt="First 6 rows of predictions"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inception-time-epilepsy-recognition/predictions_head_light.png
        style="width:100%"
    />

.. code:: python

    predictions.tail()

.. raw:: html

    <img
        id="inception-time-epilepsy-recognition-predictions-tail"
        class="blog-post-image"
        alt="Last 6 rows of predictions"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inception-time-epilepsy-recognition/predictions_tail_light.png
        style="width:100%"
    />

==========================================
Evaluation
==========================================

Finally, we calculate the classification metrics on the test set.

.. code:: python

    # calculate the classification metrics
    metrics = pd.DataFrame(columns=[c.replace("y_", "") for c in test_outputs.columns if c.startswith("y")])
    for c in metrics.columns:
        metrics[c] = {
            "Accuracy": accuracy_score(y_true=test_outputs[f"y_{c}"], y_pred=predictions[f"y_{c}"]),
            "ROC-AUC": roc_auc_score(y_true=test_outputs[f"y_{c}"], y_score=predictions[f"p_{c}"]),
            "Precision": precision_score(y_true=test_outputs[f"y_{c}"], y_pred=predictions[f"y_{c}"]),
            "Recall": recall_score(y_true=test_outputs[f"y_{c}"], y_pred=predictions[f"y_{c}"]),
            "F1": f1_score(y_true=test_outputs[f"y_{c}"], y_pred=predictions[f"y_{c}"]),
        }

We find that the model achieves a ROC-AUC score of 99.63% and an accuracy score of 97.1%
in the detection of epileptic seizures.

.. raw:: html

   <img
        id="inception-time-epilepsy-recognition-metrics"
        class="blog-post-image"
        alt="Results on Epilepsy dataset (test set)"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inception-time-epilepsy-recognition/metrics_light.png
   />

   <p class="blog-post-image-caption"> Results on Epilepsy dataset (test set).</p>

After the analysis has been completed, we can delete the model.

.. code:: python

    # delete the model
    transformer.delete_model()

.. tip::

    You can download the
    `notebook <https://github.com/fg-research/inception-time-sagemaker/blob/master/examples/Epilepsy.ipynb>`__
    with the full code from our
    `GitHub <https://github.com/fg-research/inception-time-sagemaker>`__
    repository.

******************************************
References
******************************************
[1] Chung, Y. G., Jeon, Y., Yoo, S., Kim, H., & Hwang, H. (2022).
Big data analysis and artificial intelligence in epilepsy – common data model analysis and machine learning-based seizure detection and forecasting.
*Clinical and Experimental Pediatrics*, 65(6), 272.
`doi: 10.3345/cep.2021.00766 <https://doi.org/10.3345/cep.2021.00766>`__.

[2] Ismail Fawaz, H., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D. F., Weber, J.,
Webb, G. I., Idoumghar, L., Muller, P. A., & Petitjean, F. (2020).
InceptionTime: Finding AlexNet for time series classification.
*Data Mining and Knowledge Discovery*, 34(6), 1936-1962.
`doi: 10.1007/s10618-020-00710-y <https://doi.org/10.1007/s10618-020-00710-y>`__.

[3] Villar, J. R., Vergara, P., Menéndez, M., de la Cal, E., González, V. M., & Sedano, J. (2016).
Generalized models for the classification of abnormal movements in daily life and its applicability to epilepsy convulsion recognition.
*International journal of neural systems*, 26(06), 1650037.
`doi: 10.1142/S0129065716500374 <https://doi.org/10.1142/S0129065716500374>`__.

[4] Dau, H. A., Bagnall, A., Kamgar, K., Yeh, C. C. M., Zhu, Y., Gharghabi, S., Ratanamahatana, C. A., & Keogh, E. (2019).
The UCR time series archive.
*IEEE/CAA Journal of Automatica Sinica*, 6(6), pp. 1293-1305.
`doi: 10.1109/JAS.2019.1911747 <https://doi.org/10.1109/JAS.2019.1911747>`__.


