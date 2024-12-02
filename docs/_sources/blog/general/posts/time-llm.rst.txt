.. meta::
    :thumbnail: https://fg-research.com/_static/thumbnail.png
    :description: Forecasting commodity prices with generative adversarial networks
    :keywords: Time Series, Forecasting, Generative Adversarial Network, Commodity Prices

######################################################################################
Time series forecasting with Time-LLM
######################################################################################

******************************************
Code
******************************************
To be able to run the code below, you will need to clone the
`Time-LLM original GitHub repository <https://github.com/KimMeen/Time-LLM>`__.
After that, you can run the code in a notebook or script inside
the folder where the repository was cloned.

.. note::

    Note that the code can only be run on a GPU machine.
    We used a g5.xlarge AWS EC2 instance.

==========================================
Environment Set-Up
==========================================

We start by importing all the dependencies.

.. code:: python

    # import the external modules
    import os
    import types
    import random
    import torch
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, median_absolute_error, root_mean_squared_error

    # import the internal modules
    from models.TimeLLM import Model

    # set the device
    device = torch.device("cuda:0")

After that we fix all random seed, to ensure reproducibility.

.. code:: python

    # fix all random seeds
    random_seed = 0
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

==========================================
Data Preparation
==========================================

We then load the Airline Passengers dataset in the
`Machine Learning Mastery GitHub repository <https://github.com/jbrownlee/Datasets>`__
directly into a data frame.

.. code:: python

    # load the data
    df = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/refs/heads/master/airline-passengers.csv",
        parse_dates=["Month"],
        dtype=float
    )

.. raw:: html

    <img
        id="time-llm-data"
        class="blog-post-image"
        alt="Airline Passengers dataset"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/time-llm/data_light.png
        style="width:100%"
    />

