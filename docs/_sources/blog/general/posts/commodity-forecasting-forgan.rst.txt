.. meta::
    :thumbnail: https://fg-research.com/_static/thumbnail.png
    :description: Forecasting commodity prices with generative adversarial networks
    :keywords: Time Series, Generative Adversarial Networks, Forecasting, Commodities, Finance
    :google-adsense-account: ca-pub-6940858559883413

######################################################################################
Forecasting commodity prices with generative adversarial networks
######################################################################################

.. raw:: html

    <p>
    Forecasting commodity prices is a particularly challenging task due to the intricate interplay of
    supply and demand dynamics, geopolitical factors, and market sentiment fluctuations.
    Deep learning models have been shown to be more effective than traditional statistical models at
    capturing the complex and non-linear relationships inherent in commodity markets
    <a href="#references">[1]</a>.
    </p>

    <p>
    Generative adversarial networks (GANs) <a href="#references">[2]</a>, which have led to substantial
    advancements in natural language processing and computer vision, have also found several use cases
    in the time series domain <a href="#references">[3]</a>. The application of GANs to time series is
    not restricted to data generation for augmentation or anonymization purposes, but extends to numerous
    other tasks, including, but not limited to, time series forecasting.
    </p>

    <p>
    In this post, we will focus on the ForGAN model introduced in <a href="#references">[4]</a>,
    a conditional GAN (CGAN) <a href="#references">[5]</a> for probabilistic one-step-ahead forecasting
    of univariate time series. We will implement the ForGAN model in TensorFlow, and use it for forecasting
    the daily prices of Bloomberg Commodity Index (BCOM), a leading commodities benchmark.
    </p>

    <p>
    We will download the daily close prices of Bloomberg Commodity Index from the 28<sup>th</sup> of July 2022 to
    the 26<sup>th</sup> of July 2024 from <a href="https://finance.yahoo.com" target="_blank">Yahoo! Finance</a>.
    We will train the model on the data up to the 12<sup>th</sup> of June 2024, and use the trained model to predict
    the subsequent 30 days of data up to the 26<sup>th</sup> of July 2024. We will find that the ForGAN model achieves
    a mean absolute percentage error of less than 1% over the considered 30-days period.
    </p>

******************************************
Model
******************************************
Both the generator and the discriminator of the ForGAN model are based on recurrent neural networks (RNNs).
Given that ForGAN is a CGAN, both the generator and the discriminator take as input a *condition*, which is
defined as fixed-length vector containing the most recent values of the time series, i.e. the condition
is a context window.

In the generator, the context window is passed through an RNN layer which produces an embedding vector.
After that, the embedding vector is concatenated with a noise vector, which is sampled from the standard
normal distribution. The concatenated embedding and noise vectors are then passed through a dense layer
with ReLU activation, and to a final linear output layer with a single hidden unit.
The output of the generator is the predicted next value of the time series.

In the discriminator, the context window is extended with the actual or predicted next value of the time series.
After that, the extended context window is passed through an RNN layer which produces an embedding vector.
The embedding vector is then passed to a final sigmoid output layer with a single hidden unit.
The output of the discriminator is the probability that the next value of the time series provided as input
is real (i.e. an actual value from the dataset), as opposed to synthetic (i.e. a predicted value from the generator).

.. raw:: html

    <img
        id="commodity-forecasting-forgan-diagram"
        class="blog-post-image"
        alt="ForGAN architecture"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/commodity-forecasting-forgan/diagram.png
    />

    <p class="blog-post-image-caption">ForGAN architecture.</p>

******************************************
Code
******************************************

We start by importing all the dependencies.

.. code:: python

    import warnings
    warnings.filterwarnings("ignore")

    import os
    import random
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import yfinance as yf
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

After that we define a function for fixing all random seeds, to ensure reproducibility.

.. code:: python

    def set_seeds(seed):
        '''
        Fix the random seeds.
        '''
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)


    def set_global_determinism(seed):
        '''
        Fix all sources of randomness.
        '''
        set_seeds(seed=seed)

        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

We then define the generator and discriminator models. We use LSTM layers as recurrent layers,
but GRU layers can also be used as an alternative.

.. code:: python

    class Generator(tf.keras.Model):
        '''
        Generator model.
        '''

        def __init__(self, units, noise_dimension):
            super().__init__()

            # recurrent layer
            self.rnn = tf.keras.layers.LSTM(units=units, return_sequences=False)

            # dense layer
            self.dense = tf.keras.layers.Dense(units=units + noise_dimension, activation="relu")

            # output layer
            self.out = tf.keras.layers.Dense(units=1)

        def call(self, inputs):

            # extract the inputs
            condition, noise = inputs

            # get the condition representation
            representation = self.rnn(condition)

            # extend the condition representation with the noise vector
            representation = tf.concat([representation, noise], axis=-1)

            # get the predicted value
            prediction = self.out(self.dense(representation))

            return prediction

.. code:: python

    class Discriminator(tf.keras.Model):
        '''
        Discriminator model.
        '''

        def __init__(self, units):
            super().__init__()

            # recurrent layer
            self.rnn = tf.keras.layers.LSTM(units=units, return_sequences=False)

            # output layer
            self.out = tf.keras.layers.Dense(units=1, activation="sigmoid")

        def call(self, inputs):

            # extract the inputs
            condition, next_value = inputs

            # extend the condition with the next value (either actual/real or predicted/fake)
            condition = tf.concat([condition, tf.expand_dims(next_value, axis=1)], axis=1)

            # get the condition representation
            representation = self.rnn(condition)

            # get the predicted probability
            probability = self.out(representation)

            return probability

We additionally define a custom class for training the ForGAN model and generating the probabilistic forecasts.
The class has two methods: :code:`.fit()` and :code:`.predict()`:

.. raw:: html
    <ul

    <li style="line-height: 1.75rem; margin-top: 1.75rem">The <code>.fit()</code> method scales the
    time series, splits the time series into context windows and target values, and trains the
    generator and discriminator models using standard adversarial training with the cross-entropy loss.</li>

    <li style="line-height: 1.75rem; margin-top: 1.75rem">The <code>.predict()</code> method scales
    the time series, splits the time series into context windows, and then passes the context windows
    through the generator together with different randomly generated noise vectors. Each noise vector
    results in different predictions. The predictions are transformed back to the original scale
    before being returned as an output.</li>

    </ul>

.. code:: python

    class ForGAN():
        '''
        ForGAN model.
        '''
        def __init__(self,
                     generator_units,
                     discriminator_units,
                     condition_length,
                     noise_dimension,
                     seed=42):

            self.generator_units = generator_units
            self.discriminator_units = discriminator_units
            self.condition_length = condition_length
            self.noise_dimension = noise_dimension
            self.seed = seed

        def fit(self, x, learning_rate, batch_size, epochs):

            # fix the random seeds
            set_global_determinism(seed=self.seed)

            # scale the time series
            x = x.copy().values
            self.mu = np.mean(x, axis=0)
            self.sigma = np.std(x, axis=0, ddof=1)
            x = (x - self.mu) / self.sigma

            # split the time series into condition sequences and target values
            condition = []
            target = []
            for t in range(self.condition_length, len(x)):
                condition.append(x[t - self.condition_length: t, :])
                target.append(x[t, :])
            condition = np.array(condition)
            target = np.array(target)

            # split the condition sequences and target values into batches
            dataset = tf.data.Dataset.from_tensor_slices((tf.cast(condition, tf.float32), tf.cast(target, tf.float32)))
            dataset = dataset.cache().shuffle(buffer_size=len(target), seed=self.seed).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

            # build the models
            self.generator_model = Generator(units=self.generator_units, noise_dimension=self.noise_dimension)
            self.discriminator_model = Discriminator(units=self.discriminator_units)

            # instantiate the optimizers
            generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            # define the loss function
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

            # define the training loop
            @tf.function
            def train_step(data):
                with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:

                    # extract the condition sequences and the target values
                    condition, target = data

                    # generate the noise vector
                    noise = tf.random.normal(shape=(len(condition), self.noise_dimension))

                    # generate the target values
                    prediction = self.generator_model(inputs=[condition, noise])

                    # pass the actual and the generated target values to the discriminator
                    target_probability = self.discriminator_model(inputs=[condition, target])
                    prediction_probability = self.discriminator_model(inputs=[condition, prediction])

                    # calculate the generator loss
                    generator_loss = bce(y_true=tf.ones_like(prediction_probability), y_pred=prediction_probability)

                    # calculate the discriminator loss
                    discriminator_loss = bce(y_true=tf.ones_like(target_probability), y_pred=target_probability) + \
                                         bce(y_true=tf.zeros_like(prediction_probability), y_pred=prediction_probability)

                # calculate the gradients
                generator_gradients = generator_tape.gradient(generator_loss, self.generator_model.trainable_variables)
                discriminator_gradients = discriminator_tape.gradient(discriminator_loss, self.discriminator_model.trainable_variables)

                # update the weights
                generator_optimizer.apply_gradients(zip(generator_gradients, self.generator_model.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator_model.trainable_variables))

                return generator_loss, discriminator_loss

            # train the models
            pbar = tqdm(range(epochs))
            for epoch in pbar:
                for data in dataset:
                    generator_loss, discriminator_loss = train_step(data)
                pbar.set_description_str("Epoch: {}  Generator Loss: {:.4f}  Discriminator Loss: {:.4f}".format(1 + epoch, generator_loss, discriminator_loss))

        def predict(self, x, samples):

            # fix the random seeds
            set_global_determinism(seed=self.seed)

            # scale the time series
            x = x.copy().values
            x = (x - self.mu) / self.sigma

            # split the time series into condition sequences
            condition = []
            for t in range(self.condition_length, len(x) + 1):
                condition.append(x[t - self.condition_length: t, :])
            condition = np.array(condition)

            # generate the predicted target values
            predictions = []

            # loop across the number of samples to be generated
            for _ in range(samples):

                # generate the noise vector
                noise = tf.random.normal(shape=(len(condition), self.noise_dimension))

                # generate the predicted target values
                prediction = self.generator_model(inputs=[condition, noise]).numpy()

                # transform the predicted target values back to the original scale
                prediction = self.mu + self.sigma * prediction

                # save the predicted target values
                predictions.append(prediction)

            # cast the predicted target values to array
            predictions = np.concatenate(predictions, axis=1)

            return predictions

.. raw:: html

    <p>
    Next, we download the daily close price time series of Bloomberg Commodity Index
    from the 28<sup>th</sup> of July 2022 to the 26<sup>th</sup> of July 2024 using the
    <a href="https://github.com/ranaroussi/yfinance" target="_blank">Yahoo! Finance Python API</a>.
    The dataset contains 502 daily observations.
    </p>

.. code:: python

    # download the data
    ticker = "^BCOM"
    dataset = yf.download(ticker, start="2022-07-28", end="2024-07-27")
    dataset = dataset[["Close"]].rename(columns={"Close": ticker})

.. raw:: html

    <img
        id="commodity-forecasting-forgan-time-series"
        class="blog-post-image"
        alt="Bloomberg Commodity Index from 2022-07-28 to 2024-07-26"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/commodity-forecasting-forgan/time_series_light.png
    />

    <p class="blog-post-image-caption">Bloomberg Commodity Index from 2022-07-28 to 2024-07-26.</p>

We set aside the last 30 days for testing, and use all the previous data for training.
We use a context window of 5 days, meaning that we use the last 5 prices as input to forecast the next day's price.
We set the number of hidden units of the LSTM layer equal to 256 for the generator and to 64 for the discriminator.
We set the length of the noise vectors equal to 10.
We train the model for 100 epochs with a batch size of 64 and a learning rate of 0.001.

.. code:: python

    # define the hyperparameters
    test_size = 30
    generator_units = 256
    discriminator_units = 64
    condition_length = 5
    noise_dimension = 10
    learning_rate = 0.001
    batch_size = 64
    epochs = 100

.. code:: python

    # split the data
    training_dataset = dataset.iloc[:- test_size]
    test_dataset = dataset.iloc[- test_size - condition_length: -1]

.. code:: python

    # instantiate the model
    model = ForGAN(
        generator_units=generator_units,
        discriminator_units=discriminator_units,
        condition_length=condition_length,
        noise_dimension=noise_dimension,
    )

    # train the model
    model.fit(
        x=training_dataset,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
    )

After the model has been trained, we generate the one-step-ahead predictions over the test set.
We generate 100 prices for each of the 30 days in the test set.

.. code:: python

    # generate the model predictions
    predictions = model.predict(x=test_dataset, samples=100)

.. code:: python

    predictions.shape

.. code-block:: console

    (30, 100)

We then summarize the 100 generated prices by calculating different quantiles.
For convenience, we include the actual values of the time series in the same data frame.

.. code:: python

    # summarize the model predictions
    predictions = pd.DataFrame(
        data={
            "actual": dataset.iloc[- test_size:].values.flatten(),
            "median": np.median(predictions, axis=1),
            "q005": np.quantile(predictions, 0.005, axis=1),
            "q10": np.quantile(predictions, 0.10, axis=1),
            "q90": np.quantile(predictions, 0.90, axis=1),
            "q995": np.quantile(predictions, 0.995, axis=1),
        },
        index=dataset.index[- test_size:]
    )

.. code:: python

    predictions.shape

.. code-block:: console

    (30, 6)

.. code:: python

    predictions.head()

.. raw:: html

    <img
        id="commodity-forecasting-forgan-predictions-head"
        class="blog-post-image"
        alt="First 3 rows of predictions"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/commodity-forecasting-forgan/predictions_head_light.png
    />

.. code:: python

    predictions.tail()

.. raw:: html

    <img
        id="commodity-forecasting-forgan-predictions-tail"
        class="blog-post-image"
        alt="Last 3 rows of predictions"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/commodity-forecasting-forgan/predictions_tail_light.png
    />

.. raw:: html

    <img
        id="commodity-forecasting-forgan-predictions"
        class="blog-post-image"
        alt="Actual and predicted prices from 2024-06-13 to 2024-07-26"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/commodity-forecasting-forgan/predictions_light.png
    />

    <p class="blog-post-image-caption">Actual and predicted prices over the test set (from 2024-06-13 to 2024-07-26).</p>

Finally, we calculate the root mean squared error (RMSE), mean absolute error (MAE) and
mean absolute percentage error (MAPE) of the one-step-ahead predictions over the test set.

.. note::

    Note that we use the median as point forecast.

.. code:: python

    # evaluate the model predictions
    metrics = pd.DataFrame(
        columns=["Metric", "Value"],
        data=[
            {"Metric": "RMSE", "Value": format(root_mean_squared_error(y_true=predictions["actual"], y_pred=predictions["median"]), ".4f")},
            {"Metric": "MAE", "Value": format(mean_absolute_error(y_true=predictions["actual"], y_pred=predictions["median"]), ".4f")},
            {"Metric": "MAPE", "Value": format(mean_absolute_percentage_error(y_true=predictions["actual"], y_pred=predictions["median"]), ".4f")},
        ]
    )

We find that the model achieves a MAPE of less than 1% over the test set.

.. raw:: html

    <img
        id="commodity-forecasting-forgan-metrics"
        class="blog-post-image"
        alt="Performance metrics of predicted prices over the test set (from 2024-06-13 to 2024-07-26)"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/commodity-forecasting-forgan/metrics_light.png
    />

    <p class="blog-post-image-caption">Performance metrics of predicted prices over the test set (from 2024-06-13 to 2024-07-26).</p>

.. tip::

    A Python notebook with the full code is available in our `GitHub <https://github.com/fg-research/blog/blob/master/commodity-forecasting-forgan>`__
    repository.

******************************************
References
******************************************

[1] Ben Ameur, H., Boubaker, S., Ftiti, Z., Louhichi, W., & Tissaoui, K. (2024).
Forecasting commodity prices: empirical evidence using deep learning tools. *Annals of Operations Research*, 339, pp. 349–367.
`doi: 10.1007/s10479-022-05076-6 <https://doi.org/10.1007/s10479-022-05076-6>`__.

[2] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2020).
Generative adversarial networks. *Communications of the ACM*, 63(11), pp. 139-144.
`doi: 10.1145/3422622 <https://doi.org/10.1145/3422622>`__.

[3] Brophy, E., Wang, Z., She, Q., & Ward, T. (2021).
Generative adversarial networks in time series: A survey and taxonomy. *arXiv preprint*.
`doi: 10.48550/arXiv.2107.11098 <https://doi.org/10.48550/arXiv.2107.11098>`__.

[4] Koochali, A., Schichtel, P., Dengel, A., & Ahmed, S. (2019).
Probabilistic forecasting of sensory data with generative adversarial networks – ForGAN. *IEEE Access*, 7, pp. 63868-63880.
`doi: 10.1109/ACCESS.2019.2915544 <https://doi.org/10.1109/ACCESS.2019.2915544>`__.

[5] Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets. *arXiv preprint*.
`doi: 10.48550/arXiv.1411.1784 <https://doi.org/10.48550/arXiv.1411.1784>`__.
