"""Web search through Groq's built-in browser_search tool (gpt-oss models).

The model performs the web search itself; we use ONLY the structured results it executed
(real titles, URLs and page excerpts) as candidate evidence. The model's prose answer is ignored,
so nothing it writes can become "evidence".
"""
from __future__ import annotations

import re
import time

import requests


class SearchError(RuntimeError):
    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind  # missing_key | rate_limited | timeout | http_error | bad_response | no_results | network_error


_PARA = re.compile(r"\n{1,}|(?<=[.!?])\s{2,}")


def _chunks(content: str, max_chunks: int = 8) -> list[str]:
    parts = [re.sub(r"\s+", " ", p).strip() for p in _PARA.split(content or "")]
    parts = [p for p in parts if len(p) >= 40]
    if not parts and content and len(content.strip()) >= 40:
        parts = [re.sub(r"\s+", " ", content).strip()]
    return parts[:max_chunks]


def parse_compound_response(data: dict) -> list[dict]:
    """Extract result dicts from a Groq chat-completion payload. Tolerant of missing fields."""
    if not isinstance(data, dict):
        raise SearchError("bad_response", "payload is not an object")
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SearchError("bad_response", "unexpected response envelope") from exc
    out, seen = [], set()
    for tool in message.get("executed_tools") or []:
        if not isinstance(tool, dict):
            continue
        sr = tool.get("search_results")
        results = sr.get("results") if isinstance(sr, dict) else None
        for r in results or []:
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
                "age": r.get("page_age") or r.get("date") or None,
            })
    return out


class GroqSearchClient:
    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 60.0, session=None):
        self.key, self.model, self.base, self.timeout = api_key, model, base_url, timeout
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
            raise SearchError("missing_key", "GROQ_API_KEY is not set")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a search assistant. Always use web search. Reply in one short sentence."},
                {"role": "user", "content": f"Search the web for reliable sources about: {query[:300]}"},
            ],
            "tools": [{"type": "browser_search"}],
            "tool_choice": "required",
            "temperature": 0,
            "max_tokens": 1500,  # reasoning models spend tokens thinking before/after the tool call
        }
        r = None
        for attempt in range(3):  # retry rate limits / transient server errors
            r = self._post(payload)
            if r.status_code == 429 or r.status_code >= 500:
                if attempt < 2:
                    try:
                        wait = float(r.headers.get("retry-after", 2 * (attempt + 1)))
                    except ValueError:
                        wait = 2 * (attempt + 1)
                    time.sleep(min(wait, 8))
                    continue
            break
        if r.status_code == 429:
            raise SearchError("rate_limited", "Groq rate limit reached")
        if r.status_code in (401, 403):
            raise SearchError("http_error", f"authentication failed ({r.status_code}); check GROQ_API_KEY")
        if r.status_code >= 400:
            detail = ""
            try:
                detail = str(r.json().get("error", {}).get("message", ""))[:200]
            except Exception:
                pass
            raise SearchError("http_error", f"{r.status_code} {detail}".strip())
        try:
            results = parse_compound_response(r.json())
        except ValueError as exc:
            raise SearchError("bad_response", "invalid JSON") from exc
        if not results:
            # Make "model didn't search / returned nothing" visible instead of looking like success.
            raise SearchError("no_results", "search tool returned no results")
        return results[: max(1, count)]