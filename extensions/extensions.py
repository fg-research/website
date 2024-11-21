
def google_analytics(app, pagename, templatename, context, doctree):
    metatags = context.get('metatags', '')
    metatags += """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-2L0F07XRQM"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
    
      gtag('config', 'G-2L0F07XRQM');
    </script>
    """
    metatags += """
    <script type="text/javascript">
    var _iub = _iub || [];
    _iub.csConfiguration = {
        "floatingPreferencesButtonDisplay": false,
        "askConsentAtCookiePolicyUpdate":true,
        "countryDetection":true,
        "enableFadp":true,
        "enableLgpd":true,
        "enableUspr":true,
        "lang":"en",
        "lgpdAppliesGlobally":false,
        "logViaAlert":true,
        "perPurposeConsent":true,
        "preferenceCookie":{"expireAfter":1},
        "siteId":3844372,
        "storage":{"useSiteId":true},
        "cookiePolicyId":49690203,
        "i18n":{"en":{"banner":{
            "dynamic":{
            "body": "fg-research.com and selected third parties use cookies or similar technologies for technical purposes and, with your consent, for functionality, experience, measurement and marketing (personalized ads). Use the “Accept” button to consent. Use the “Reject” button to continue without accepting."}}}},
            "banner":{
            "acceptButtonCaptionColor":"#0F172A",
            "acceptButtonColor":"#D3D4D5,
            "rejectButtonCaptionColor":"#0F172A",
            "rejectButtonColor":"#D3D4D5",
            "customizeButtonCaptionColor":"#ffffff",
            "customizeButtonColor":"#64748B",
            "customizeButtonDisplay":true,
            "rejectButtonDisplay":true,
            "acceptButtonDisplay":true,
            "backgroundColor":"#FFFFFF",
            "closeButtonDisplay":false,
            "explicitWithdrawal":true,
            "fontSizeBody":"12px",
            "listPurposes":true,
            "linksColor":"#0F172A",
            "ownerName":"fg-research.com",
            "position":"float-bottom-center",
            "showPurposesToggles":false,
            "showTitle":false,
            "showTotalNumberOfProviders":true,
            "slideDown":false,
            "textColor":"#0F172A",
            "acceptButtonCaption":"Accept",
            "customizeButtonCaption":"Learn more",
            "rejectButtonCaption":"Reject"
            }
        };
    </script>
    <script type="text/javascript" src="https://cs.iubenda.com/autoblocking/3844372.js"></script>
    <script type="text/javascript" src="//cdn.iubenda.com/cs/gpp/stub.js"></script>
    <script type="text/javascript" src="//cdn.iubenda.com/cs/iubenda_cs.js" charset="UTF-8" async></script>
    """
    context['metatags'] = metatags


def setup(app):
    app.connect('html-page-context', google_analytics)

