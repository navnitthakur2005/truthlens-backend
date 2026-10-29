"""Source-quality indicators.

These are coarse *priors based on the type of publisher*, NOT a verdict on any specific page.
A high-tier source can be wrong and a low-tier one can be right; the verifier is told to weigh the
actual content and to treat tiers only as a hint. Meanings are exposed to the UI via QUALITY_LEGEND.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

QUALITY_LEGEND = {
    "official": "Government, intergovernmental or public-agency website (primary source for its own domain).",
    "scientific": "University, research institution, scientific publisher or space/health agency.",
    "news": "Established news organisation or wire service with editorial standards.",
    "factcheck": "Dedicated fact-checking organisation (secondary analysis of claims).",
    "reference": "General reference / user-edited encyclopaedia; useful for orientation, not primary evidence.",
    "unrated": "No reliability information available for this domain. Judge by content.",
}

_GOV = re.compile(r"(^|\.)(gov|mil)(\.[a-z]{2})?$|(^|\.)gob\.[a-z]{2}$|(^|\.)nic\.in$|(^|\.)gouv\.[a-z]{2}$|"
                  r"(^|\.)govt\.[a-z]{2}$|(^|\.)gc\.ca$|(^|\.)europa\.eu$")
_EDU = re.compile(r"(^|\.)edu(\.[a-z]{2})?$|(^|\.)ac\.[a-z]{2}$")

_OFFICIAL = {"who.int", "un.org", "worldbank.org", "imf.org", "oecd.org", "isro.gov.in", "pib.gov.in"}
_SCIENTIFIC = {"nature.com", "science.org", "thelancet.com", "nejm.org", "bmj.com", "pubmed.ncbi.nlm.nih.gov",
               "arxiv.org", "esa.int", "cern.ch", "sciencedirect.com", "springer.com", "cell.com", "pnas.org"}
_NEWS = {"reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "afp.com", "bloomberg.com", "ft.com", "theguardian.com",
         "nytimes.com", "washingtonpost.com", "wsj.com", "npr.org", "pbs.org", "aljazeera.com", "dw.com",
         "thehindu.com", "indianexpress.com", "hindustantimes.com", "timesofindia.indiatimes.com", "ndtv.com",
         "economist.com", "cbc.ca", "abc.net.au", "france24.com", "japantimes.co.jp", "straitstimes.com"}
_FACTCHECK = {"snopes.com", "politifact.com", "factcheck.org", "fullfact.org", "boomlive.in", "altnews.in",
              "factcheck.afp.com", "leadstories.com", "checkyourfact.com", "vishvasnews.com"}
_REFERENCE = {"wikipedia.org", "britannica.com"}

_TRACKING = re.compile(r"^(utm_|fbclid|gclid|mc_|ref$|ref_)")


def registrable_host(url: str) -> str:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def _match(host: str, domains: set[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def classify_source(url: str) -> dict:
    host = registrable_host(url)
    tier = "unrated"
    if _match(host, _FACTCHECK):
        tier = "factcheck"
    elif _GOV.search(host) or _match(host, _OFFICIAL):
        tier = "official"
    elif _EDU.search(host) or _match(host, _SCIENTIFIC):
        tier = "scientific"
    elif _match(host, _NEWS):
        tier = "news"
    elif _match(host, _REFERENCE):
        tier = "reference"
    return {"tier": tier, "domain": host, "meaning": QUALITY_LEGEND[tier]}


TIER_WEIGHT = {"official": 1.0, "scientific": 0.9, "news": 0.7, "factcheck": 0.7, "reference": 0.4, "unrated": 0.2}


def normalize_url(url: str) -> str:
    """Canonical form used for de-duplication (drop fragment, tracking params, www, trailing slash)."""
    from urllib.parse import parse_qsl, urlencode, urlunparse

    p = urlparse(url)
    q = [(k, v) for k, v in parse_qsl(p.query) if not _TRACKING.match(k.lower())]
    host = registrable_host(url)
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme.lower(), host, path, "", urlencode(q), ""))
