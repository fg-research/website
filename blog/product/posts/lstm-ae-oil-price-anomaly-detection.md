----
orphan: true
----

# Detecting Anomalies in Financial Time Series with the LSTM-AE SageMaker Algorithm 
Anomaly detection in financial time series plays a crucial role in identifying unusual market conditions 
that could affect trading strategies and pose risks to investors.
Anomaly detection can help identify abnormal price movements or trading volumes associated with specific events, 
such as earnings announcements, release of economic indicators, or geopolitical tensions.
Anomaly detection algorithms are also useful for automatically detecting and correcting data quality issues in 
the market data time series used for calculating portfolio risk measures [[1]](#references).

Recurrent autoencoders are standard deep learning architectures for detecting anomalies in sequential data.
The autoencoder is trained in an unsupervised manner to learn a low-dimensional latent representation of the data (through the encoder), 
and to reconstruct the normal behavior of the data from this latent representation (through the decoder).
The trained autoencoder is then applied to new unseen data, and an anomaly is detected whenever 
the observed data deviates significantly from the autoencoder's reconstruction.

Different autoencoder architectures have been proposed in the literature on time series anomaly detection [[2]](#references).
In this post, we will focus on the [Long Short Term Memory Networks based Encoder-Decoder scheme for Anomaly Detection](https://doi.org/10.48550/arXiv.1607.00148) [[3]](#references), 
which we will refer to as the LSTM-AE model. We will demonstrate how to use our [Amazon SageMaker](https://aws.amazon.com/sagemaker/) implementation of the LSTM-AE model, 
the [LSTM-AE SageMaker algorithm](https://aws.amazon.com/marketplace/pp/prodview-up2haipz3j472), 
for detecting anomalies in oil price time series. 

We will download the West Texas Intermediate (WTI) and Brent daily price time series from the [Federal Reserve Economic Data (FRED) database](https://fred.stlouisfed.org/). 
After that we will train the LSTM-AE model on the data up to the 1<sup>st</sup> of August 2019, 
and use the trained model to reconstruct the subsequent data up to the 31<sup>st</sup> of December 2020. 
We will then show how the LSTM-AE model detects the abnormal oil prices observed at the end of April 2020 during the COVID-19 pandemic.  

## Model
The encoder and decoder of the LSTM-AE model consist of a single LSTM layer and have the same number of hidden units.
The encoder takes as input the time series and returns the hidden states.
The hidden states of the encoder are used for initializing the hidden states of the decoder, which reconstructs the time series in reversed order. 

<img src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/diagrams/lstm-ae-architecture.png style="width:80%"/>

*LSTM-AE architecture (source: [doi: 10.48550/arXiv.1607.00148](https://doi.org/10.48550/arXiv.1607.00148))*

The autoencoder parameters are learned on a training set containing only normal data (i.e. without anomalies) by minimizing the mean squared error (MSE) between the actual and reconstructed values of the time series.
After the model has been trained, a Gaussian distribution is fitted to the model's reconstruction errors on an independent validation set (also without anomalies) using Maximum Likelihood Estimation (MLE). 

At inference time, the model reconstructs the values of all the time series (which can now include anomalies) 
and calculates the squared Mahalanobis distance between the reconstruction errors and the Gaussian distribution previously estimated on normal data. 
The computed squared Mahalanobis distance is then used as an anomaly score: the larger the squared Mahalanobis distance at a given a time step, the more likely the time step is to be an anomaly.

## Data
We use the [Python API for FRED](https://github.com/mortada/fredapi) for downloading the data.
If you don't have an API key, you can request one at [this link](http://api.stlouisfed.org/api_key.html).

```python
from fredapi import Fred

fred = Fred(api_key_file="api_key.txt")
```

We download the data from the 20<sup>th</sup> of May 1987, which is the start date of the Brent time series, to the 31<sup>st</sup> of December 2020.
The downloaded dataset contains 8,772 daily price observations. 

```python
import pandas as pd

dataset = pd.DataFrame({
    "WTI": fred.get_series("DCOILWTICO", observation_start="1987-05-20", observation_end="2020-12-31"),
    "BRENT": fred.get_series("DCOILBRENTEU", observation_start="1987-05-20", observation_end="2020-12-31")
})
```

On the 20<sup>th</sup> of April 2020, the WTI price decreased from \$18.31 to -\$36.98, going negative for the first time in its history,
while on the next day the 21<sup>st</sup> of April 2020, the Brent price decreased from \$17.36 to \$9.12.

<img src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/oil-price-anomaly-detection/prices.png sizes="(max-width: 1440) 100%, 80%"/>

*WTI and Brent daily prices from 1987-05-20 to 2020-12-31.*

We use the percentage changes in the daily prices (or daily returns) for training the LSTM-AE model.

```python
dataset = dataset.pct_change().fillna(value=0)
```

The percentage change in WTI price on the 20<sup>th</sup> of April 2020 was -302%, 
followed by a -124% decrease on the 21<sup>st</sup> of April 2020 and 
a 53% increase on the 22<sup>nd</sup> of April 2020.
The percentage change in Brent price on the 20<sup>th</sup> of April 2020 was -12%, 
followed by a -47% decrease on the 21<sup>st</sup> of April 2020 and 
a 51% increase on the 22<sup>nd</sup> of April 2020.

<img src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/oil-price-anomaly-detection/returns.png />

*WTI and Brent daily returns from 1987-05-20 to 2020-12-31.*

## Code

### Environment Set-Up
We start by setting up the SageMaker environment.

:::{note}
To be able to run the code below, you need to have an active subscription to the algorithm.
You can subscribe to a free trial from the [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-up2haipz3j472) in order to get your Amazon Resource Name (ARN).
:::

```python
import sagemaker

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
```

### Data Preparation
After that we split the data into training and test sets, which we save to S3 in CSV format.
We use the first 8,402 observations for training, and the remaining 370 observations for testing.
The training set covers the time window from 20<sup>th</sup> of May 1987 to the 1<sup>st</sup> of August 2019, 
while the test set covers the time window from the 2<sup>nd</sup> of August 2019 to the 31<sup>st</sup> of December 2020.

```python
# define the train-test split cutoff
cutoff = 8402

# split the data
training_dataset = dataset.iloc[:cutoff]
test_dataset = dataset.iloc[cutoff:]

# save the training data in S3
training_data = sagemaker_session.upload_string_as_file_body(
    body=training_dataset.to_csv(index=False, header=False),
    bucket=bucket,
    key="oil_price_train.csv"
)

# save the test data in S3
test_data = sagemaker_session.upload_string_as_file_body(
    body=test_dataset.to_csv(index=False, header=False),
    bucket=bucket,
    key="oil_price_test.csv"
)
```

### Training
We can now run a training job on the training dataset.

:::{note}
The algorithm uses the first 80% of the training dataset for learning the LSTM parameters, 
and the remaining 20% of the training dataset for estimating the Gaussian distribution parameters.
:::

```python
# create the estimator
estimator = sagemaker.algorithm.AlgorithmEstimator(
    algorithm_arn=algo_arn,
    role=role,
    instance_count=1,
    instance_type=instance_type,
    input_mode="File",
    sagemaker_session=sagemaker_session,
    hyperparameters={
        "sequence-length": 10,
        "sequence-stride": 1,
        "hidden-size": 100,
        "lr": 0.001,
        "batch-size": 32,
        "epochs": 500,
    },
)

# run the training job
estimator.fit({"training": training_data})
```
 
### Inference
Once the training job has completed, we can run a batch transform job on the test dataset.

```python
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
```

The results of the batch transform job are saved in an output file in S3 with the same name as the input file and with the `".out"` file extension.
The output file contains the anomaly scores in the first column, and the reconstructed values of the time series in the subsequent columns.

:::{note}
The LSTM-AE is a multivariate time series anomaly detection model and, therefore, it generates only one anomaly score for all time series at each time step.
:::

```python
# load the model outputs from S3
reconstructions = sagemaker_session.read_s3_file(
    bucket=bucket,
    key_prefix=f"{transformer.latest_transform_job.name}/oil_price_test.csv.out"
)

# convert the model outputs to data frame
reconstructions = pd.read_csv(io.StringIO(reconstructions), header=None, dtype=float)
```

After loading the anomaly scores and the reconstructions from S3, we can visualize the results.

<img src=https://fg-research-blog.s3.eu-west-1.amazonaws.com/oil-price-anomaly-detection/results.png />

*LSTM-AE reconstructions and anomaly score for WTI and Brent daily returns from 2019-08-02 to 2020-12-31.*

We find that, as expected, the anomaly score exhibits the largest upward spikes on the 20<sup>th</sup> (anomaly score = 810,274),
21<sup>st</sup> (anomaly score = 64,522) and 22<sup>nd</sup> (anomaly score = 15,533) of April 2020.

You can download the [notebook](https://github.com/fg-research/lstm-ae-sagemaker/blob/master/example/oil_price_anomaly_detection.ipynb) 
with the full code from our [GitHub](https://github.com/fg-research/lstm-ae-sagemaker) repository.

## References
[1] Crépey, S., Lehdili, N., Madhar, N., & Thomas, M. (2022). Anomaly Detection in Financial Time Series by Principal Component Analysis and Neural Networks. *Algorithms*, 15(10), 385, [doi: 10.3390/a15100385](https://doi.org/10.3390/a15100385).

[2] Darban, Z. Z., Webb, G. I., Pan, S., Aggarwal, C. C., & Salehi, M. (2022). Deep learning for time series anomaly detection: A survey. *arXiv preprint*, [doi: 10.48550/arXiv.2211.05244](https://doi.org/10.48550/arXiv.2211.05244).

[3] Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection. *arXiv preprint*, [doi: 10.48550/arXiv.1607.00148](https://doi.org/10.48550/arXiv.1607.00148).