
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
        "emailMarketing":{
            "styles":{
                "backgroundColor":"#FFFFFF",
                "buttonsBackgroundColor":"#3B82F6",
                "buttonsTextColor":"#FFFFFF",
                "footerBackgroundColor":"#64748B",
                "footerTextColor":"#FFFFFF",
                "textColor":"#0F172A"
            }
        },
        "enableFadp":true,
        "enableLgpd":true,
        "enableTcf":false,
        "enableUspr":true,
        "googleAdditionalConsentMode":true,
        "lang":"en",
        "lgpdAppliesGlobally":false,
        "perPurposeConsent":true,
        "siteId":3844372,
        "storage":{"useSiteId":true},
        "tcfPurposes":{"2":"consent_only","7":"consent_only","8":"consent_only","9":"consent_only","10":"consent_only","11":"consent_only"},
        "cookiePolicyId":49690203,
        "banner":{
            "acceptButtonColor":"#0F172A",
            "acceptButtonDisplay":true,
            "backgroundColor":"#FFFFFF",
            "closeButtonDisplay":false,
            "customizeButtonColor":"#64748B",
            "customizeButtonDisplay":true,
            "explicitWithdrawal":true,
            "fontSizeBody":"12px",
            "listPurposes":true,
            "linksColor":"#0F172A",
            "ownerName":"fg-research.com",
            "position":"float-bottom-center",
            "rejectButtonColor":"#0F172A",
            "rejectButtonDisplay":true,
            "showPurposesToggles":false,
            "showTitle":false,
            "showTotalNumberOfProviders":false,
            "textColor":"#0F172A"
            }
        };
    </script>
    <script type="text/javascript" src="https://cs.iubenda.com/autoblocking/3844372.js"></script>
    <script type="text/javascript" src="//cdn.iubenda.com/cs/tcf/stub-v2.js"></script>
    <script type="text/javascript" src="//cdn.iubenda.com/cs/tcf/safe-tcf-v2.js"></script>
    <script type="text/javascript" src="//cdn.iubenda.com/cs/gpp/stub.js"></script>
    <script type="text/javascript" src="//cdn.iubenda.com/cs/iubenda_cs.js" charset="UTF-8" async></script>
    """
    context['metatags'] = metatags


def setup(app):
    app.connect('html-page-context', google_analytics)

