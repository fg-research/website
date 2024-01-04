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
            f.childNodes[1].childNodes[0].src = '_static/AWSMP_NewLogo_RGB_BLK.png'
            f.childNodes[1].href = '_static/AWSMP_NewLogo_RGB_BLK.png'
        }else if(window.localStorage.getItem('darkMode') == 'dark' || getPreferredColorScheme() == 'dark'){
            f.childNodes[1].childNodes[0].src = '_static/AWSMP_NewLogo_RGB_WHT.png'
            f.childNodes[1].href = '_static/AWSMP_NewLogo_RGB_WHT.png'
        };
        if(screen.width > 1440){
            f.childNodes[1].childNodes[0].style.width = '35%'
        }else{
            f.childNodes[1].childNodes[0].style.width = '55%'
        };
    };

    var a = document.getElementsByClassName('reference external');
    if (a !== null) {
        for(let i = 0; i < a.length; i++){
            a[i].target = '_blank'
        };
    };

}, 1);

