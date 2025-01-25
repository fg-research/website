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
`Time-LLM GitHub repository <https://github.com/KimMeen/Time-LLM>`__.
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

After that we fix all random seeds, to ensure reproducibility.

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
Data
==========================================

We load the Airline Passengers dataset in the
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
    />

We now define the context length and prediction length. We use the previous 24 months to predict the subsequent 12 months.

.. code:: python

    # define the context length
    seq_len = 24

    # define the prediction length
    pred_len = 12

We set aside the last 12 months for testing, and use all the remaining data for training.

.. code:: python

    # extract the training data
    df_train = df[["Passengers"]].iloc[:- pred_len]

    # extract the test data
    df_test = df[["Passengers"]].iloc[- (seq_len + pred_len):]

We scale the data by subtracting the mean and dividing by the standard deviation.

.. code:: python

    # calculate the scaling parameters on the training data
    mu = df_train.mean(axis=0).item()
    sigma = df_train.std(axis=0, ddof=1).item()

    # scale the training and test data
    df_train = (df_train - mu) / sigma
    df_test = (df_test - mu) / sigma

Finally, we split the data into sequences.

.. code:: python

    # define a function for splitting the data into sequences
    def get_sequences(df, seq_len, pred_len):
        x = []
        y = []
        for t in range(seq_len, len(df) - pred_len + 1):
            x.append(df.iloc[t - seq_len: t].values)
            y.append(df.iloc[t: t + pred_len].values)
        x = np.array(x)
        y = np.array(y)
        return x, y

    # generate the training sequences
    x_train, y_train = get_sequences(df_train, seq_len, pred_len)

    # generate the test sequences
    x_test, y_test = get_sequences(df_test, seq_len, pred_len)

==========================================
Model
==========================================
---------------------------
Training
---------------------------

We start by creating the model.

.. code:: python

    # create the model
    model = Model(
        configs=types.SimpleNamespace(
            prompt_domain=True,
            content="Monthly totals of a airline passengers from USA, from January 1949 through December 1960.",
            task_name="short_term_forecast",  # not used
            enc_in=None,  # not used
            pred_len=pred_len,
            seq_len=seq_len,
            llm_model="LLAMA",
            llm_dim=4096,
            llm_layers=1,
            d_model=32,
            d_ff=32,
            patch_len=16,
            stride=8,
            n_heads=4,
            dropout=0,
        )
    )
    model.to(torch.bfloat16)
    model.to(device)

After that, we train the model for 40 epochs with a batch size of 8 and a learning rate of 0.001.

.. code:: python

    # define the training parameters
    batch_size = 8
    lr = 0.001
    epochs = 40

    # create the training dataset
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float()
    )

    # create the training dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # instantiate the optimizer
    optimizer = torch.optim.Adam(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    # train the model
    model.train()
    for epoch in range(epochs):
        losses = []
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            yhat = model(
                x_enc=x,
                x_mark_enc=None,
                x_dec=None,
                x_mark_dec=None
            )
            loss = torch.nn.functional.mse_loss(yhat, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print(f'epoch: {format(1 + epoch, ".0f")} loss: {format(np.mean(losses), ",.8f")}')

---------------------------
Inference
---------------------------
After the model has been trained, we can use it for generating the forecasts over the test set.

.. code:: python

    # generate the forecasts
    model.eval()
    yhat_test = model(
        x_enc=torch.from_numpy(x_test).float().to(device),
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None
    ).detach().cpu().numpy().flatten()

    # transform the forecasts back to the original scale
    yhat_test = mu + sigma * yhat_test