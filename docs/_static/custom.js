window.setInterval(function(){
    var f = document.getElementById('aws-marketplace-logo');
    if(window.localStorage.getItem('darkMode') == 'light'){
        f.childNodes[1].src = '_static/AWSMP_NewLogo_RGB_BLK.png'
    }else{
        f.childNodes[1].src = '_static/AWSMP_NewLogo_RGB_WHT.png'
    };
    if(screen.width > 1440){
        f.childNodes[1].style.width = '35%'
    }else{
        f.childNodes[1].style.width = '50%'
    };
}, 1);

