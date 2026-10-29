"""SSRF-safe page fetching and readable-text extraction.

Protections: http/https only, no credentials in URL, ports 80/443 only, every hostname is resolved
and ALL addresses must be public (no loopback/private/link-local/reserved/multicast), redirects are
followed manually and re-validated (max 3), response size and time are capped, only text content
types are accepted, robots.txt is honoured. After connecting we additionally verify the peer IP on a
best-effort basis to narrow DNS-rebinding windows.
"""
from __future__ import annotations

import ipaddress
import json
import re
import socket
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

ALLOWED_PORTS = {None, 80, 443}
MAX_REDIRECTS = 3


class UnsafeURL(ValueError):
    pass


class FetchError(RuntimeError):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def _ip_is_public(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip.split("%")[0])
    except ValueError:
        return False
    if getattr(a, "ipv4_mapped", None):
        a = a.ipv4_mapped
    return a.is_global and not (a.is_multicast or a.is_loopback or a.is_link_local or a.is_private or a.is_reserved)


def validate_url(url: str, resolver=socket.getaddrinfo) -> str:
    """Raise UnsafeURL unless the URL is safe to request. Returns the URL."""
    if not isinstance(url, str) or len(url) > 2048:
        raise UnsafeURL("invalid URL")
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise UnsafeURL("only http/https URLs are allowed")
    if p.username or p.password:
        raise UnsafeURL("credentials in URL are not allowed")
    host = p.hostname
    if not host:
        raise UnsafeURL("missing host")
    try:
        port = p.port
    except ValueError as exc:
        raise UnsafeURL("invalid port") from exc
    if port not in ALLOWED_PORTS:
        raise UnsafeURL(f"port {port} is not allowed")
    if host.lower() in ("localhost",) or host.lower().endswith((".local", ".internal", ".localhost")):
        raise UnsafeURL("local hostnames are not allowed")
    # literal IPs are checked directly (never trust a resolver for them); hostnames are resolved
    try:
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        literal = None
    if literal is not None and not _ip_is_public(str(literal)):
        raise UnsafeURL("host is a non-public IP address")
    try:
        infos = resolver(host, port or (443 if p.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeURL(f"cannot resolve host: {exc}") from exc
    addrs = {i[4][0] for i in infos}
    if not addrs or not all(_ip_is_public(a) for a in addrs):
        raise UnsafeURL("host resolves to a non-public address")
    return url


@dataclass
class Page:
    url: str
    title: str = ""
    text: str = ""
    published: str | None = None
    paragraphs: list = field(default_factory=list)


_DATE_META = ("article:published_time", "og:article:published_time", "date", "pubdate", "publishdate",
              "dc.date", "dc.date.issued", "datepublished", "parsely-pub-date", "citation_publication_date")


def _find_date(soup: BeautifulSoup) -> str | None:
    for m in soup.find_all("meta"):
        key = (m.get("property") or m.get("name") or m.get("itemprop") or "").lower()
        if key in _DATE_META and m.get("content"):
            return m["content"].strip()[:40]
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(s.string or "")
        except Exception:
            continue
        for node in data if isinstance(data, list) else [data]:
            if isinstance(node, dict):
                v = node.get("datePublished") or node.get("dateCreated")
                if isinstance(v, str):
                    return v[:40]
    t = soup.find("time", attrs={"datetime": True})
    return t["datetime"][:40] if t else None


def extract_readable(html: str, url: str = "") -> Page:
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    date = _find_date(soup)
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "iframe"]):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    paras = [re.sub(r"\s+", " ", p.get_text(" ", strip=True)) for p in root.find_all(["p", "li", "h1", "h2", "h3"])]
    paras = [p for p in paras if len(p) >= 40]
    return Page(url=url, title=title, text="\n".join(paras), published=date, paragraphs=paras)


class SafeFetcher:
    def __init__(self, timeout: float = 8.0, max_bytes: int = 1_500_000,
                 user_agent: str = "TruthLensBot/1.0", respect_robots: bool = True, session=None,
                 resolver=socket.getaddrinfo):
        self.timeout, self.max_bytes, self.ua = timeout, max_bytes, user_agent
        self.respect_robots = respect_robots
        self.session = session or requests.Session()
        self.resolver = resolver
        self._robots: dict[str, RobotFileParser | None] = {}

    # --- low-level -----------------------------------------------------------
    def _get(self, url: str, accept: str) -> tuple[requests.Response, bytes]:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            validate_url(current, self.resolver)
            try:
                r = self.session.get(current, timeout=self.timeout, stream=True, allow_redirects=False,
                                     headers={"User-Agent": self.ua, "Accept": accept})
            except requests.Timeout as exc:
                raise FetchError("timeout", current) from exc
            except requests.RequestException as exc:
                raise FetchError("network_error", type(exc).__name__) from exc
            self._check_peer(r)
            if r.is_redirect or r.status_code in (301, 302, 303, 307, 308):
                loc = r.headers.get("Location")
                r.close()
                if not loc:
                    raise FetchError("bad_redirect")
                current = urljoin(current, loc)
                continue
            if r.status_code in (401, 403, 451):
                r.close()
                raise FetchError("access_denied", str(r.status_code))
            if r.status_code == 429:
                r.close()
                raise FetchError("rate_limited")
            if r.status_code >= 400:
                r.close()
                raise FetchError("http_error", str(r.status_code))
            ctype = r.headers.get("Content-Type", "").lower()
            if accept != "*/*" and not any(t in ctype for t in ("text/html", "text/plain", "application/xhtml")):
                r.close()
                raise FetchError("unsupported_content_type", ctype)
            chunks, size = [], 0
            try:
                for chunk in r.iter_content(65536):
                    size += len(chunk)
                    if size > self.max_bytes:
                        break  # truncate rather than fail: the useful text is usually at the top
                    chunks.append(chunk)
            except requests.RequestException as exc:
                raise FetchError("network_error", type(exc).__name__) from exc
            finally:
                r.close()
            return r, b"".join(chunks)
        raise FetchError("too_many_redirects")

    @staticmethod
    def _check_peer(r) -> None:
        try:  # best effort: confirm the connected socket address is public
            sock = r.raw._connection.sock  # type: ignore[attr-defined]
            ip = sock.getpeername()[0]
            if not _ip_is_public(ip):
                r.close()
                raise UnsafeURL("connected to a non-public address")
        except UnsafeURL:
            raise
        except Exception:
            pass

    # --- robots.txt ----------------------------------------------------------
    def allowed_by_robots(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        if origin not in self._robots:
            rp = RobotFileParser()
            try:
                _, body = self._get(origin + "/robots.txt", "*/*")
                rp.parse(body.decode("utf-8", "ignore").splitlines())
                self._robots[origin] = rp
            except (FetchError, UnsafeURL):
                self._robots[origin] = None  # no readable robots.txt -> allowed
        rp = self._robots[origin]
        return True if rp is None else rp.can_fetch(self.ua, url)

    # --- public --------------------------------------------------------------
    def fetch_page(self, url: str) -> Page:
        try:
            validate_url(url, self.resolver)
        except UnsafeURL as exc:
            raise FetchError("unsafe_url", str(exc)) from exc
        try:
            if not self.allowed_by_robots(url):
                raise FetchError("blocked_by_robots")
            r, body = self._get(url, "text/html,application/xhtml+xml,text/plain;q=0.8")
        except UnsafeURL as exc:
            raise FetchError("unsafe_url", str(exc)) from exc
        enc = r.encoding or "utf-8"
        html = body.decode(enc if enc.lower() != "iso-8859-1" else "utf-8", "ignore")
        page = extract_readable(html, url)
        if len(page.text) < 80:
            raise FetchError("no_readable_text")
        return page
