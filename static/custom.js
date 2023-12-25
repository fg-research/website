window.setInterval(function(){
    console.log(window.localStorage.getItem('darkMode'));
    var f = document.getElementById('aws-marketplace-logo');
    if(window.localStorage.getItem('darkMode') == 'light'){
        f.childNodes[1].href = 'https://aws.amazon.com/marketplace/seller-profile?id=seller-nkd47o2qbdvb2'
        f.childNodes[1].childNodes[0].src = '_static/AWSMP_NewLogo_RGB_BLK.png'
    }else{
        f.childNodes[1].href = 'https://aws.amazon.com/marketplace/seller-profile?id=seller-nkd47o2qbdvb2'
        f.childNodes[1].childNodes[0].src = '_static/AWSMP_NewLogo_RGB_WHT.png'
    };
    if(screen.width > 1440){
        f.childNodes[1].childNodes[0].style.width = '33%'
    }else{
        f.childNodes[1].childNodes[0].style.width = '40%'
    }
}, 1);

