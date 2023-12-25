wget https://raw.githubusercontent.com/fg-research/cfc-tsf-sagemaker/master/README.MD -O ./algorithms/forecasting/cfc.md
wget https://raw.githubusercontent.com/fg-research/lstm-ad-sagemaker/master/README.MD -O ./algorithms/anomaly-detection/lstm-ad.md
wget https://raw.githubusercontent.com/fg-research/lstm-ae-sagemaker/master/README.MD -O ./algorithms/anomaly-detection/lstm-ae.md
wget https://raw.githubusercontent.com/fg-research/cnn-kmeans-sagemaker/master/README.MD -O ./algorithms/clustering/cnn-kmeans.md
wget https://raw.githubusercontent.com/fg-research/cnn-svc-sagemaker/master/README.MD -O ./algorithms/classification/cnn-svc.md
wget https://raw.githubusercontent.com/fg-research/lstm-fcn-sagemaker/master/README.MD -O ./algorithms/classification/lstm-fcn.md
wget https://raw.githubusercontent.com/fg-research/inception-time-sagemaker/master/README.MD -O ./algorithms/classification/inception-time.md
make html
git add --all
git commit -m "update website"
git push origin master