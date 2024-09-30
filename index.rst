.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Train, tune and deploy state-of-the-art machine learning models for time series in Amazon SageMaker
   :keywords: Amazon SageMaker, Time Series, Machine Learning, Forecasting, Anomaly Detection

########################################################
Advanced Time Series Solutions in Amazon SageMaker
########################################################
.. rst-class:: lead

    Train, tune and deploy state-of-the-art machine learning models for time series in Amazon SageMaker

.. _fg_research_logo:

.. image:: /static/background.png
  :align: left
  :alt: background image
  :width: 90%

.. _aws_marketplace_logo:

.. image:: /static/AWSMP_NewLogo_RGB_BLK.png
   :align: right
   :alt: AWS Marketplace Logo

******************************************
Overview
******************************************
We provide Amazon SageMaker algorithms for multiple time series tasks, including forecasting, anomaly detection, classification and clustering.
Each algorithm implements a state-of-the-art machine learning model designed specifically for time series.

******************************************
Features
******************************************
Automated Data Handling
   The algorithms work directly on raw time series data in CSV format. All the required data preprocessing and scaling is performed internally by the algorithm's code.

Automatic Model Tuning
   The algorithms support `automatic model tuning <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html>`__ for optimizing the model hyperparameters in order to achieve the best possible performance on a given dataset.

Incremental Training
    Most of the algorithms support `incremental training <https://docs.aws.amazon.com/sagemaker/latest/dg/incremental-training.html>`__ to continue training the model on the same dataset or to fine-tune the model on a different dataset.

Accelerated Training
   The algorithms were built by extending the latest `deep learning containers <https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers-prebuilt.html>`__ and support both CPU and GPU training. Most of the algorithms also support multi-GPU training.

*****************************************
Documentation
*****************************************
.. raw:: html

    <p>Each algorithm has a dedicated <a href="https://github.com/fg-research" target="_blank">GitHub</a> repository with detailed documentation and step-by-step tutorials in Jupyter notebook format. Several use cases are also discussed in our <a href="blog/product/index.html#product" target="_blank">blog</a>.</p>

******************************************
Pricing
******************************************
.. raw:: html

    <p>The algorithms are available on the <a href="https://aws.amazon.com/marketplace/seller-profile?id=seller-nkd47o2qbdvb2" target="_blank">AWS Marketplace</a> on a usage-based pricing plan. Each algorithm offers a 5 days free trial.</p>

******************************************
Support
******************************************
For support, contact `support@fg-research.com <mailto:support@fg-research.com>`__.

.. raw:: html

    <p style="margin-bottom: 1rem"> <br/> </p>

------

.. grid:: 3

    .. grid-item::
        :columns: 5

        .. toctree::
           :caption: Algorithms
           :maxdepth: 1

           algorithms/time-series-forecasting/index
           algorithms/time-series-anomaly-detection/index
           algorithms/time-series-classification/index
           algorithms/time-series-clustering/index

    .. grid-item::
        :columns: 3

        .. toctree::
           :caption: Blog
           :maxdepth: 1

           blog/product/index
           blog/general/index

    .. grid-item::
        :columns: 4

        .. toctree::
           :caption: Terms and Conditions
           :maxdepth: 1

           terms/disclaimer/index
           terms/eula/index
