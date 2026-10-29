"""Web search through the Tavily Search API.

Tavily returns structured results (title, URL, page excerpt). We use ONLY those results as candidate
evidence, so nothing a language model writes can become "evidence". Searching costs no Groq tokens.
"""
from __future__ import annotations

import re
import time

import requests


class SearchError(RuntimeError):
    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind  # missing_key | rate_limited | timeout | http_error | bad_response | no_results | network_error


_PARA = re.compile(r"\n{1,}|\s*\[\.\.\.\]\s*|(?<=[.!?])\s{2,}")


def _chunks(content: str, max_chunks: int = 8) -> list[str]:
    parts = [re.sub(r"\s+", " ", p).strip() for p in _PARA.split(content or "")]
    parts = [p for p in parts if len(p) >= 40]
    if not parts and content and len(content.strip()) >= 40:
        parts = [re.sub(r"\s+", " ", content).strip()]
    return parts[:max_chunks]


def parse_tavily_response(data: dict) -> list[dict]:
    """Extract result dicts from a Tavily /search payload. Tolerant of missing fields."""
    if not isinstance(data, dict):
        raise SearchError("bad_response", "payload is not an object")
    results = data.get("results")
    if not isinstance(results, list):
        raise SearchError("bad_response", "missing 'results' list")
    out, seen = [], set()
    for r in results:
        if not isinstance(r, dict):
            continue
        url, title = r.get("url"), r.get("title")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        content = r.get("content") if isinstance(r.get("content"), str) else ""
        out.append({
            "title": (title or url) if isinstance(title, str) else url,
            "url": url,
            "snippets": _chunks(content),
            "age": r.get("published_date") or r.get("page_age") or None,
        })
    return out


class TavilySearchClient:
    def __init__(self, api_key: str, base_url: str = "https://api.tavily.com/search",
                 depth: str = "basic", timeout: float = 20.0, session=None):
        self.key, self.base, self.depth, self.timeout = api_key, base_url, depth, timeout
        self.session = session or requests.Session()

    @property
    def configured(self) -> bool:
        return bool(self.key)

    def _post(self, payload: dict) -> requests.Response:
        try:
            return self.session.post(self.base, json=payload, timeout=self.timeout,
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        except requests.Timeout as exc:
            raise SearchError("timeout") from exc
        except requests.RequestException as exc:
            raise SearchError("network_error", type(exc).__name__) from exc

    def search(self, query: str, count: int = 6) -> list[dict]:
        if not self.key:
            raise SearchError("missing_key", "TAVILY_API_KEY is not set")
        payload = {
            "query": query[:400],
            "search_depth": self.depth,  # "basic" = 1 credit, "advanced" = 2 credits (more excerpt text)
            "max_results": max(1, min(count, 10)),
            "include_answer": False,  # we never use Tavily's generated answer, only the sources
            "include_raw_content": False,
        }
        r = None
        for attempt in range(2):  # one retry for rate limits / transient server errors
            r = self._post(payload)
            if (r.status_code == 429 or r.status_code >= 500) and attempt == 0:
                time.sleep(2)
                continue
            break
        if r.status_code in (429, 432, 433):  # 432/433 = plan or key usage limit reached
            raise SearchError("rate_limited", f"Tavily limit reached ({r.status_code})")
        if r.status_code in (401, 403):
            raise SearchError("http_error", f"authentication failed ({r.status_code}); check TAVILY_API_KEY")
        if r.status_code >= 400:
            detail = ""
            try:
                body = r.json()
                err = body.get("detail", body.get("error", ""))
                detail = str(err.get("error", err) if isinstance(err, dict) else err)[:200]
            except Exception:
                pass
            raise SearchError("http_error", f"{r.status_code} {detail}".strip())
        try:
            results = parse_tavily_response(r.json())
        except ValueError as exc:
            raise SearchError("bad_response", "invalid JSON") from exc
        if not results:
            raise SearchError("no_results", "search returned no results")
        return results[: max(1, count)]