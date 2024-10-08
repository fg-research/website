function getPreferredColorScheme() {
  if (window.matchMedia) {
    if(window.matchMedia('(prefers-color-scheme: dark)').matches){
      return 'dark';
    } else {
      return 'light';
    }
  }
  return 'light';
}

window.setInterval(function(){

    var f = document.getElementById('aws-marketplace-logo');
    if (f !== null) {
        if(window.localStorage.getItem('darkMode') == 'light'){
            f.src = '_static/AWSMP_NewLogo_RGB_BLK.png'
        }else if(window.localStorage.getItem('darkMode') == 'dark' || getPreferredColorScheme() == 'dark'){
            f.src = '_static/AWSMP_NewLogo_RGB_WHT.png'
        };
    };

    var elements = [
        'lstm-ae-oil-price-anomaly-detection-toctree',
        'lstm-fcn-ecg-classification-toctree',
        'fred-md-overview-toctree',
        'lstm-ad-ecg-anomaly-detection-toctree',
        'inflation-forecasting-random-forest-toctree',
        'lnn-equity-forecasting-toctree',
        'commodity-forecasting-forgan-toctree',
        'rnn-fx-forecasting-toctree',
        'cnn-kmeans-ecg-clustering-toctree',
        'inception-time-epilepsy-recognition-toctree',
        'equity-trend-prediction-automl-toctree',
        'electricity-forecasting-chronos-toctree'
    ];
    for (var i = 0; i < elements.length; i++) {
        var f = document.getElementById(elements[i]);
        if (f !== null) {
            if(window.localStorage.getItem('darkMode') == 'light'){
                f.childNodes[1].childNodes[1].childNodes[0].style = 'color: #0f172a !important'
            }else if(window.localStorage.getItem('darkMode') == 'dark' || getPreferredColorScheme() == 'dark'){
                f.childNodes[1].childNodes[1].childNodes[0].style = 'color: #e1e7ef !important'
            };
        };
    };

    var elements = [
        'lstm-ae-oil-price-anomaly-detection-prices',
        'lstm-ae-oil-price-anomaly-detection-returns',
        'lstm-ae-oil-price-anomaly-detection-results-plot',
        'lstm-ae-oil-price-anomaly-detection-results-table',
        'lstm-fcn-ecg-classification-class-distribution',
        'lstm-fcn-ecg-classification-metrics',
        'cnn-kmeans-ecg-clustering-time-series',
        'cnn-kmeans-ecg-clustering-results',
        'lstm-ad-ecg-anomaly-detection-dataset',
        'lstm-ad-ecg-anomaly-detection-results',
        'fred-md-dataset-head',
        'fred-md-dataset-tail',
        'inflation-forecasting-random-forest-diagram',
        'inflation-forecasting-random-forest-time-series',
        'inflation-forecasting-random-forest-forecasts-plot',
        'inflation-forecasting-random-forest-forecasts-table-head',
        'inflation-forecasting-random-forest-forecasts-table-tail',
        'inflation-forecasting-random-forest-targets-table-head',
        'inflation-forecasting-random-forest-targets-table-tail',
        'inflation-forecasting-random-forest-errors-table',
        'lnn-equity-forecasting-time-series',
        'lnn-equity-forecasting-predictions',
        'lnn-equity-forecasting-dataset-head',
        'lnn-equity-forecasting-dataset-tail',
        'lnn-equity-forecasting-renamed-dataset-head',
        'lnn-equity-forecasting-renamed-dataset-tail',
        'lnn-equity-forecasting-forecasts',
        'lnn-equity-forecasting-metrics',
        'commodity-forecasting-forgan-time-series',
        'commodity-forecasting-forgan-predictions',
        'commodity-forecasting-forgan-predictions-head',
        'commodity-forecasting-forgan-predictions-tail',
        'commodity-forecasting-forgan-metrics',
        'rnn-fx-forecasting-time-series',
        'rnn-fx-forecasting-predictions',
        'rnn-fx-forecasting-dataset-head',
        'rnn-fx-forecasting-dataset-tail',
        'rnn-fx-forecasting-renamed-dataset-head',
        'rnn-fx-forecasting-renamed-dataset-tail',
        'rnn-fx-forecasting-returns',
        'rnn-fx-forecasting-metrics',
        'inception-time-epilepsy-recognition-time-series',
        'inception-time-epilepsy-recognition-metrics',
        'inception-time-epilepsy-recognition-training-dataset-head',
        'inception-time-epilepsy-recognition-training-dataset-tail',
        'inception-time-epilepsy-recognition-test-inputs-head',
        'inception-time-epilepsy-recognition-test-inputs-tail',
        'inception-time-epilepsy-recognition-test-outputs-head',
        'inception-time-epilepsy-recognition-test-outputs-tail',
        'inception-time-epilepsy-recognition-predictions-head',
        'inception-time-epilepsy-recognition-predictions-tail',
        'equity-trend-prediction-automl-dataset-head',
        'equity-trend-prediction-automl-dataset-tail',
        'equity-trend-prediction-automl-importances',
        'equity-trend-prediction-automl-matrix',
        'equity-trend-prediction-automl-metrics',
        'equity-trend-prediction-automl-predictions-head',
        'equity-trend-prediction-automl-predictions-tail',
        'equity-trend-prediction-automl-raw-predictions-head',
        'equity-trend-prediction-automl-raw-predictions-tail',
        'equity-trend-prediction-automl-roc-auc',
        'equity-trend-prediction-automl-time-series',
        'electricity-forecasting-chronos-time-series',
        'electricity-forecasting-chronos-sarima-forecasts',
        'electricity-forecasting-chronos-sarima-forecasts-head',
        'electricity-forecasting-chronos-sarima-forecasts-tail',
        'electricity-forecasting-chronos-sarima-metrics',
        'electricity-forecasting-chronos-chronos-forecasts',
        'electricity-forecasting-chronos-chronos-forecasts-head',
        'electricity-forecasting-chronos-chronos-forecasts-tail',
        'electricity-forecasting-chronos-chronos-metrics'
    ];
    for (var i = 0; i < elements.length; i++) {
        var f = document.getElementById(elements[i]);
        if (f !== null) {
            if(window.localStorage.getItem('darkMode') == 'light'){
                f.src = f.src.replace('dark', 'light')
            }else if(window.localStorage.getItem('darkMode') == 'dark' || getPreferredColorScheme() == 'dark'){
                f.src = f.src.replace('light', 'dark')
            };
        };
    };

}, 1);

