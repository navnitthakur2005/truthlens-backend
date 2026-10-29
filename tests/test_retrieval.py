import pytest
import requests

from app.retrieval import evidence as ev
from app.retrieval.brave import BraveClient, SearchError, parse_brave_response
from app.retrieval.fetcher import FetchError, SafeFetcher, UnsafeURL, extract_readable, validate_url
from app.retrieval.sources import classify_source, normalize_url
from tests.conftest import FakeBrave, FakeFetcher
from app.retrieval.fetcher import Page


def public_resolver(host, port, type=None):
    return [(2, 1, 6, "", ("93.184.216.34", port))]


def test_parse_brave_response_ok_and_tolerant():
    data = {"web": {"results": [
        {"title": "<strong>A</strong> &amp; B", "url": "https://x.org/a", "description": "d1", "extra_snippets": ["e1"], "page_age": "2023-01-01"},
        {"title": "no url"}, "junk", {"url": "https://y.org", "title": "T"}]}}
    out = parse_brave_response(data)
    assert len(out) == 2 and out[0]["title"] == "A & B" and out[0]["snippets"] == ["d1", "e1"]
    assert parse_brave_response({}) == []
    with pytest.raises(SearchError):
        parse_brave_response({"web": "oops"})
    with pytest.raises(SearchError):
        parse_brave_response([])


_BAD = object()


class Resp:
    def __init__(self, status=200, payload=None):
        self.status_code, self._p = status, payload if payload is not None else {}
    def json(self):
        if self._p is _BAD: raise ValueError("invalid json")
        return self._p


class Sess:
    def __init__(self, resp=None, exc=None): self.resp, self.exc = resp, exc
    def get(self, *a, **k):
        if self.exc: raise self.exc
        return self.resp


@pytest.mark.parametrize("sess,kind", [
    (Sess(Resp(429)), "rate_limited"), (Sess(Resp(401)), "http_error"), (Sess(Resp(500)), "http_error"),
    (Sess(exc=requests.Timeout()), "timeout"), (Sess(exc=requests.ConnectionError()), "network_error"),
    (Sess(Resp(200, _BAD)), "bad_response")])
def test_brave_client_errors(sess, kind):
    with pytest.raises(SearchError) as e:
        BraveClient("k", "http://x", session=sess).search("q")
    assert e.value.kind == kind


def test_brave_client_missing_key_and_success():
    with pytest.raises(SearchError) as e:
        BraveClient("", "http://x").search("q")
    assert e.value.kind == "missing_key"
    ok = BraveClient("k", "http://x", session=Sess(Resp(200, {"web": {"results": [{"title": "t", "url": "https://a.org"}]}})))
    assert ok.search("q")[0]["url"] == "https://a.org"


@pytest.mark.parametrize("url", ["http://127.0.0.1/", "http://169.254.169.254/", "http://[::1]/", "file:///etc/passwd",
                                 "ftp://example.com", "http://u:p@example.com", "http://localhost", "https://example.com:8443/",
                                 "http://192.168.1.1/", "http://metadata.internal/"])
def test_ssrf_blocked(url):
    with pytest.raises(UnsafeURL):
        validate_url(url)


def test_ssrf_hostname_resolving_to_private_blocked():
    def evil(host, port, type=None): return [(2, 1, 6, "", ("10.0.0.7", port))]
    with pytest.raises(UnsafeURL):
        validate_url("https://innocent.example.com/", evil)
    assert validate_url("https://example.com/x", public_resolver)


def test_fetcher_redirect_to_internal_is_blocked():
    class R:
        status_code, headers, is_redirect = 302, {"Location": "http://127.0.0.1/admin"}, True
        raw = None
        def close(self): pass
    class S:
        def get(self, *a, **k): return R()
    f = SafeFetcher(session=S(), respect_robots=False, resolver=public_resolver)
    with pytest.raises(FetchError) as e:
        f.fetch_page("https://example.com/")
    assert e.value.reason == "unsafe_url"


def test_extract_readable_and_date():
    html = """<html><head><title>T</title><meta property="article:published_time" content="2023-08-24T10:00:00Z"></head>
    <body><nav>menu menu menu</nav><article><p>Chandrayaan-3 soft-landed near the lunar south pole on 23 August 2023 as planned.</p>
    <p>short</p><script>alert(1)</script></article></body></html>"""
    p = extract_readable(html, "https://x.org")
    assert p.title == "T" and p.published.startswith("2023-08-24")
    assert len(p.paragraphs) == 1 and "alert" not in p.text and "menu" not in p.text


def test_source_classification_and_normalization():
    assert classify_source("https://www.isro.gov.in/x")["tier"] == "official"
    assert classify_source("https://www.nasa.gov/x")["tier"] == "official"
    assert classify_source("https://news.mit.edu/a")["tier"] == "scientific"
    assert classify_source("https://www.reuters.com/a")["tier"] == "news"
    assert classify_source("https://www.snopes.com/a")["tier"] == "factcheck"
    assert classify_source("https://en.wikipedia.org/wiki/X")["tier"] == "reference"
    assert classify_source("https://random.example.net")["tier"] == "unrated"
    assert normalize_url("https://www.X.org/a/?utm_source=t&id=2#frag") == normalize_url("https://x.org/a?id=2")


CLAIMS = [{"id": "C1", "text": "Chandrayaan-3 landed on the Moon in 2024.", "queries": ["q1", "q2"]}]


def test_search_all_dedups_and_collects_errors():
    res, errs = ev.search_all(FakeBrave(), CLAIMS, 6)
    assert len(res) == 2 and errs == []   # both queries return same URLs -> de-duplicated
    with pytest.raises(SearchError):
        ev.search_all(FakeBrave(error=SearchError("rate_limited")), CLAIMS, 6)


def test_build_evidence_fetch_failure_falls_back_to_snippets():
    res, _ = ev.search_all(FakeBrave(), CLAIMS, 6)
    chosen = ev.rank_results(res, CLAIMS, 5)
    assert chosen[0]["quality"]["tier"] == "official"     # authority outranks the blog
    evidence, notes = ev.build_evidence(chosen, CLAIMS, FakeFetcher())
    assert evidence and all(not e["retrieval"]["page_fetched"] for e in evidence)
    assert evidence[0]["passages"][0]["origin"] == "search_snippet"
    assert notes and evidence[0]["id"] == "E1"


def test_build_evidence_uses_page_text_and_drops_irrelevant():
    res, _ = ev.search_all(FakeBrave(), CLAIMS, 6)
    chosen = ev.rank_results(res, CLAIMS, 5)
    page = Page(url=chosen[0]["url"], title="ISRO", published="2023-08-24",
                paragraphs=["Chandrayaan-3 soft-landed on the Moon on 23 August 2023 near the south pole.",
                            "Cookie banner text unrelated to anything at all in the claim."])
    evidence, _ = ev.build_evidence(chosen, CLAIMS, FakeFetcher({chosen[0]["url"]: page}))
    first = evidence[0]
    assert first["retrieval"]["page_fetched"] and first["published"] == "2023-08-24"
    assert len(first["passages"]) == 1 and first["passages"][0]["origin"] == "page"
