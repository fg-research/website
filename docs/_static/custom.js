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
        'liquid-neural-networks-overview-toctree',
        'fred-md-overview-toctree',
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
        'lstm-ae-oil-price-anomaly-detection-results-table'
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

