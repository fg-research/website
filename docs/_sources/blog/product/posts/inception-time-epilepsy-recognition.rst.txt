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

.. raw:: html

    <img
        id="inception-time-epilepsy-recognition-diagram"
        class="blog-post-image"
        alt="Inception block."
        style="width:100%"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/inception-time-epilepsy-recognition/diagram.png
    />

    <p class="blog-post-image-caption">Inception block.</p>


******************************************
Data
******************************************

.. raw:: html

    <p>
    We use the Epilepsy dataset introduced in <a href="#references">[...]</a> and available
    in the UCR Time Series Classification Archive <a href="#references">[...]</a>.

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
    You can subscribe to a free trial from the `AWS Marketplace <https://aws.amazon.com/marketplace/pp/prodview-3hdblqdz5nx4m>`__
    in order to get your Amazon Resource Name (ARN). In this post we use version 1.8 of the InceptionTime SageMaker algorithm,
    which runs in the PyTorch 2.1.0 Python 3.10 deep learning container.


==========================================
Data Preparation
==========================================


.. warning::

    To be able to run the code below, you need to download the datasets (`"ECG200_TRAIN.txt"` and `"ECG200_TEST.txt"`)
    from the `UCR Time Series Classification Archive <http://www.timeseriesclassification.com/description.php?Dataset=ECG200>`__
    and store them in the SageMaker notebook instance.



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
        style="width:85%"
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


[6] Dau, H. A., Bagnall, A., Kamgar, K., Yeh, C. C. M., Zhu, Y., Gharghabi, S., Ratanamahatana, C. A., & Keogh, E. (2019).
The UCR time series archive.
*IEEE/CAA Journal of Automatica Sinica*, vol. 6, no. 6, pp. 1293-1305.
`doi: 10.1109/JAS.2019.1911747 <https://doi.org/10.1109/JAS.2019.1911747>`__.


