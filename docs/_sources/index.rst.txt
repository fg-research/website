.. meta::
   :description: Time Series Analysis in Amazon SageMaker.
   :keywords: Amazon Web Services, AWS, Amazon SageMaker, Time Series, Machine Learning

########################################################
Time Series Analysis in Amazon SageMaker
########################################################
.. rst-class:: lead

Train, tune and deploy state-of-the-art deep learning models for time series in Amazon SageMaker.

.. _fg_research_logo:

.. figure:: /static/logo.png
  :align: left
  :width: 70%
  :alt: fg-research logo

.. _aws_marketplace_logo:

.. figure:: /static/AWSMP_NewLogo_RGB_BLK.png
   :align: right
   :width: 35%
   :alt: AWS Marketplace Logo

******************************************
Product
******************************************
We provide Amazon SageMaker algorithms for multiple time series tasks, including forecasting, anomaly detection, clustering and classification.
Each algorithm implements a state-of-the-art deep learning architecture designed specifically for time series.

******************************************
Features
******************************************
Automated Data Handling
   The algorithms work directly on raw time series data in CSV format. All the required data preprocessing and scaling is performed internally by our code.

Automatic Model Tuning
   The algorithms support `automatic model tuning <https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html>`_ for optimizing the neural network hyperparameters in order to achieve the best possible performance on a given dataset.

Incremental Training
    Most of the algorithms support `incremental training <https://docs.aws.amazon.com/sagemaker/latest/dg/incremental-training.html>`_ which can be used to continue training the model on the same dataset or to fine-tune the model on a different dataset.

Accelerated Training
   The algorithms were built by extending the latest `deep learning containers <https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers-prebuilt.html>`_ and support both CPU and GPU training. Most of the algorithms also support multi-GPU training.

*****************************************
Documentation
*****************************************
Each algorithm has a dedicated `GitHub <https://github.com/fg-research>`_ repository with detailed documentation and step-by-step tutorials in Jupyter notebook format.

******************************************
Pricing
******************************************
The algorithms are available on the `AWS Marketplace <https://aws.amazon.com/marketplace/seller-profile?id=seller-nkd47o2qbdvb2>`_
on a usage-based pricing plan. Each algorithm offers a 5 days free trial.

******************************************
Support
******************************************
For support, contact `support@fg-research.com <mailto:support@fg-research.com>`_.

.. toctree::
   :caption: Algorithms
   :hidden:

   algorithms/time-series-forecasting/index
   algorithms/time-series-anomaly-detection/index
   algorithms/time-series-clustering/index
   algorithms/time-series-classification/index

.. toctree::
   :caption: Blog
   :hidden:

   blog/product/index
   blog/general/index
