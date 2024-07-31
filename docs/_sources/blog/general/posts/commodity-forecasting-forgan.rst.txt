.. meta::
   :thumbnail: https://fg-research.com/_static/thumbnail.png
   :description: Forecasting commodity prices with generative adversarial networks
   :keywords: Time Series, Generative Adversarial Networks, Forecasting, Commodities

######################################################################################
Forecasting commodity prices with generative adversarial networks
######################################################################################

.. raw:: html

    <p>
    Forecasting commodity prices is a particularly challenging task due to the intricate interplay of
    supply and demand dynamics, geopolitical factors, and market sentiment fluctuations.
    Deep learning models have been shown to be more effective than traditional statistical models at
    capturing the complex and non-linear relationships inherent in commodity price data
    <a href="#references">[1]</a>.
    </p>

    <p>
    Generative Adversarial Networks (GANs), which have led to substantial advancements in natural
    language processing and computer visions, have also found several applications in time series
    analysis <a href="#references">[2]</a>.
    </p>

******************************************
Model
******************************************

.. raw:: html

    <img
        id="commodity-forecasting-forgan-diagram"
        class="blog-post-image"
        alt="ForGAN architecture"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/commodity-forecasting-forgan/diagram_light.png
    />

    <p class="blog-post-image-caption">ForGAN architecture.</p>

******************************************
Data
******************************************

.. raw:: html

    <img
        id="commodity-forecasting-forgan-time-series"
        class="blog-post-image"
        alt="Bloomberg Commodity Index from 2022-07-28 to 2024-07-26"
        src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/commodity-forecasting-forgan/time_series_light.png
    />

    <p class="blog-post-image-caption">Bloomberg Commodity Index from 2022-07-28 to 2024-07-26.</p>


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

We then define the generator and discriminator architectures.

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

We also define a custom class for training the model and generating the distributional forecasts.

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

            # define the loss functions
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

            # get the condition sequence
            condition = np.expand_dims(x[- self.condition_length:], axis=0)

            # generate the next value of the target time series
            simulation = []

            # loop across the number of samples to be generated
            for sample in range(samples):

                # generate the noise vector
                noise = tf.random.normal(shape=(len(condition), self.noise_dimension))

                # generate the next target value
                prediction = self.generator_model(inputs=[condition, noise]).numpy()

                # transform the generated target values back to the original scale
                prediction = self.mu + self.sigma * prediction

                # save the generated target values
                simulation.append(prediction)

            # cast the generated target values to array
            simulation = np.concatenate(simulation, axis=1)

            return simulation

.. raw:: html

    <p>
    Next, we download the daily close price time series of Bloomberg Commodity Index
    from the 28<sup>th</sup> of July 2022 to the 26<sup>th</sup> of July 2024 using the
    <a href="https://github.com/ranaroussi/yfinance" target="_blank">Yahoo! Finance Python API</a>.
    The dataset contains 502 daily observations.
    </p>

.. code:: python

    ticker = "^BCOM"

.. code:: python

    dataset = yf.download(ticker, start="2022-07-28", end="2024-07-27")
    dataset = dataset[["Close"]].rename(columns={"Close": ticker})

We set aside the last 30 days for testing, and use all the previous data for training.

.. code:: python

    test_size = 30

.. code:: python

    model = ForGAN(
        generator_units=256,
        discriminator_units=64,
        condition_length=5,
        noise_dimension=10,
        seed=42
    )

    model.fit(
        x=dataset.iloc[:- test_size],
        learning_rate=0.001,
        batch_size=64,
        epochs=100,
    )

After the model has been trained, we generate the one-step-ahead predictions over the test set.
We use the model for generating 100 prices for each of the 30 days in the test set.

.. code:: python

    simulations = []
    for t in reversed(range(1, 1 + test_size)):
        simulations.append(model.predict(x=dataset.iloc[:- t], samples=100))
    simulations = np.concatenate(simulations, axis=0)

We then summarize the 100 generated prices by calculating the median and the quantiles.
For convenience, we include the actual values of the time series in the same data frame.

.. code:: python

    predictions = pd.DataFrame(
        data={
            "actual": dataset.iloc[- test_size:].values.flatten(),
            "median": np.median(simulations, axis=1),
            "q005": np.quantile(simulations, 0.005, axis=1),
            "q995": np.quantile(simulations, 0.995, axis=1),
            "q10": np.quantile(simulations, 0.10, axis=1),
            "q90": np.quantile(simulations, 0.90, axis=1),
        },
        index=dataset.index[-test_size:]
    )

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
Note that we use the median as point forecast.

.. code:: python

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

[2] Brophy, E., Wang, Z., She, Q., & Ward, T. (2021).
Generative adversarial networks in time series: A survey and taxonomy. *arXiv preprint*.
`doi: 10.48550/arXiv.2107.11098 <https://doi.org/10.48550/arXiv.2107.11098>`__.

[3] Koochali, A., Schichtel, P., Dengel, A., & Ahmed, S. (2019).
Probabilistic forecasting of sensory data with generative adversarial networks – ForGAN. *IEEE Access*, 7, pp. 63868-63880.
`doi: 10.1109/ACCESS.2019.2915544 <https://doi.org/10.1109/ACCESS.2019.2915544>`__.

[4] Vuletić, M., Prenzel, F., & Cucuringu, M. (2024).
Fin-GAN: Forecasting and classifying financial time series via generative adversarial networks.
*Quantitative Finance*, 24(2), pp. 175-199.
`doi: 10.1080/14697688.2023.2299466 <https://doi.org/10.1080/14697688.2023.2299466>`__.
