
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
    _iub.csConfiguration = {"askConsentAtCookiePolicyUpdate":true,"countryDetection":true,"emailMarketing":{"styles":{"backgroundColor":"#FFFFFF","buttonsBackgroundColor":"#3B82F6","buttonsTextColor":"#FFFFFF","footerBackgroundColor":"#64748B","footerTextColor":"#FFFFFF","textColor":"#0F172A"}},"enableFadp":true,"enableLgpd":true,"enableUspr":true,"gdprAppliesGlobally":false,"lang":"en","lgpdAppliesGlobally":false,"perPurposeConsent":true,"siteId":3844372,"storage":{"useSiteId":true},"cookiePolicyId":49690203,"banner":{"acceptButtonColor":"#3B82F6","acceptButtonDisplay":true,"backgroundColor":"#FFFFFF","closeButtonDisplay":false,"customizeButtonColor":"#64748B","customizeButtonDisplay":true,"explicitWithdrawal":true,"fontSizeBody":"12px","listPurposes":true,"linksColor":"#0F172A","ownerName":"fg-research.com","position":"float-bottom-center","rejectButtonColor":"#3B82F6","rejectButtonDisplay":true,"showTitle":false,"showTotalNumberOfProviders":true,"textColor":"#0F172A"}};
    </script>
    <script type="text/javascript" src="https://cs.iubenda.com/autoblocking/3844372.js"></script>
    <script type="text/javascript" src="//cdn.iubenda.com/cs/gpp/stub.js"></script>
    <script type="text/javascript" src="//cdn.iubenda.com/cs/iubenda_cs.js" charset="UTF-8" async></script>
    """
    context['metatags'] = metatags


def setup(app):
    app.connect('html-page-context', google_analytics)

