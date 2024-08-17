.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Epilepsy convulsion recognition with the InceptionTime SageMaker Algorithm
   :keywords: Amazon SageMaker, Time Series, Classification, Convolutional Neural Network, Epilepsy

######################################################################################
Epilepsy convulsion recognition with the InceptionTime SageMaker Algorithm
######################################################################################

.. raw:: html

    <p>
    ...
    </p>


******************************************
Model
******************************************
InceptionTime is an ensemble model. The only difference between the models in the ensemble
is in the initial values of the weights, which are sampled from the Glorot uniform distribution,
while the model architecture and hyperparameters are the same.

Each model consists of a stack of Inception blocks.
Each block includes three convolutional layers with kernel sizes of 10, 20 and 40 and a max pooling layer.
The block input is processed by the four layers in parallel, and the four outputs are concatenated
before being passed to a batch normalization layer followed by a ReLU activation.

.. raw:: html

    <img
        id="inception-time-epilepsy-recognition-diagram"
        class="blog-post-image"
        alt="Inception block."
        style="width:100%"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inception-time-epilepsy-recognition/diagram.png
    />

    <p class="blog-post-image-caption">Inception block.</p>

A residual connection is applied between the input time series and the output of the second block,
and after that between every three blocks.
The residual connection processes the inputs using an additional convolutional layer with a kernel size of 1
followed by a batch normalization layer.
The processed inputs are then added to the output, which is transformed by a ReLU activation.
The output of the last block is passed to an average pooling layer which removes the time dimension,
and then to a final linear layer.

At inference time, the class probabilities predicted by the different models are averaged in order to obtain
a unique predicted probability and, therefore, a unique predicted label, for each class.

******************************************
Data
******************************************
.. raw:: html

    <p>
    We use the Epilepsy dataset introduced in <a href="#references">[...]</a> and available
    in the UCR Time Series Classification Archive <a href="#references">[...]</a>.
    The data was collected from 6 study participants who conducted 4 different activities
    while wearing a tri-axial accelerometer on their wrist: walking, running, sewing and
    mimicking epileptic seizures.
    The mimicked epileptic seizures followed a protocol defined by a medical expert.
    </p>

    <p>
    The dataset contains 275 3-dimensional time series of length 206 recorded at a
    sampling frequency of 16 Hz, corresponding to approximately 13 seconds.
    137 time series are included in the training set, while the remaining
    138 time series are included in the test set.
    The training set and test time series correspond to different participants
    (3 participants are included in the training set,
    while the remaining 3 participants are included in the test set).
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

==========================================
Environment Set-Up
==========================================

We start by importing all the requirements and setting up the SageMaker environment.

.. warning::

    To be able to run the code below, you need to have an active subscription to the InceptionTime SageMaker algorithm.
    You can subscribe to a free trial from the `AWS Marketplace <https://aws.amazon.com/marketplace/pp/prodview-omz7rumnllmla>`__
    in order to get your Amazon Resource Name (ARN). In this post we use version 1.8 of the InceptionTime SageMaker algorithm,
    which runs in the PyTorch 2.1.0 Python 3.10 deep learning container.

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

.. warning::

    To be able to run the code below, you need to download the data
    from the `UCR Time Series Classification Archive <http://www.timeseriesclassification.com/description.php?Dataset=ECG200>`__
    and store the :code:`ARFF` files in the SageMaker notebook instance.

After that we define a function for reading and preparing the data
in the format required by the InceptionTime SageMaker algorithm.
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

.. raw:: html

    <img
        id="inception-time-epilepsy-recognition-training-dataset-head"
        class="blog-post-image"
        alt="First 6 rows of training dataset"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inception-time-epilepsy-recognition/training_dataset_head_light.png
        style="width:100%"
    />


.. raw:: html

    <img
        id="inception-time-epilepsy-recognition-training-dataset-tail"
        class="blog-post-image"
        alt="Last 6 rows of training dataset"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inception-time-epilepsy-recognition/training_dataset_tail_light.png
        style="width:100%"
    />

.. raw:: html

    <img
        id="inception-time-epilepsy-recognition-test-inputs-head"
        class="blog-post-image"
        alt="First 6 rows of test inputs"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inception-time-epilepsy-recognition/test_inputs_head_light.png
    />


.. raw:: html

    <img
        id="inception-time-epilepsy-recognition-test-inputs-tail"
        class="blog-post-image"
        alt="Last 6 rows of test inputs"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inception-time-epilepsy-recognition/test_inputs_tail_light.png
    />

.. raw:: html

    <img
        id="inception-time-epilepsy-recognition-test-outputs-head"
        class="blog-post-image"
        alt="First 6 rows of test outputs"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inception-time-epilepsy-recognition/test_outputs_head_light.png
    />


.. raw:: html

    <img
        id="inception-time-epilepsy-recognition-test-outputs-tail"
        class="blog-post-image"
        alt="Last 6 rows of test outputs"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inception-time-epilepsy-recognition/test_outputs_tail_light.png
    />

==========================================
Training
==========================================

Now that the training dataset is available in an accessible S3 bucket, we are ready to fit the model.



==========================================
Inference
==========================================

Once the training job has completed, we can run a batch transform job on the test dataset.


The results are saved in an output file in S3 with the same name as the input file and with the `".out"` file extension.
The results include the predicted cluster labels, which are stored in the first column, and the extracted features,
which are stored in the subsequent columns.


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

[] Villar, J. R., Vergara, P., Menéndez, M., de la Cal, E., González, V. M., & Sedano, J. (2016).
Generalized models for the classification of abnormal movements in daily life and its applicability to epilepsy convulsion recognition.
*International journal of neural systems*, 26(06), 1650037.
`doi: 10.1142/S0129065716500374 <https://doi.org/10.1142/S0129065716500374>`__.

[] Dau, H. A., Bagnall, A., Kamgar, K., Yeh, C. C. M., Zhu, Y., Gharghabi, S., Ratanamahatana, C. A., & Keogh, E. (2019).
The UCR time series archive.
*IEEE/CAA Journal of Automatica Sinica*, 6(6), pp. 1293-1305.
`doi: 10.1109/JAS.2019.1911747 <https://doi.org/10.1109/JAS.2019.1911747>`__.


