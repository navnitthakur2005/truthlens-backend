"""Search -> rank -> fetch -> passage extraction -> de-duplication."""
from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor

from app.retrieval.tavily_search import SearchError
from app.retrieval.fetcher import FetchError, SafeFetcher
from app.retrieval.sources import TIER_WEIGHT, classify_source, normalize_url

_STOP = set("""a an the and or but if of in on at to for from by with as is are was were be been being it its this that these
those he she they we you i his her their our your not no than then so such into over after before about also has have had
will would can could may might says said say new more most""".split())
_TOK = re.compile(r"[a-z0-9]+")
MAX_PASSAGES = 3
PASSAGE_CHARS = 450
MIN_RELEVANCE = 0.25


def tokens(s: str) -> set[str]:
    return {t for t in _TOK.findall(s.lower()) if t not in _STOP and (len(t) > 2 or t.isdigit())}


def overlap(claim_tokens: set[str], text: str) -> float:
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & tokens(text)) / len(claim_tokens)


def trim(s: str, n: int = PASSAGE_CHARS) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= n:
        return s
    cut = s[:n]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > n * 0.6 else cut).rstrip(",;:- ") + "…"


def select_passages(paragraphs: list[str], claim_texts: list[str], limit: int = MAX_PASSAGES) -> list[tuple[str, float]]:
    ctoks = [tokens(c) for c in claim_texts]
    scored = []
    for p in paragraphs:
        score = max((overlap(ct, p) for ct in ctoks), default=0.0)
        if score >= MIN_RELEVANCE:
            scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    return [(trim(p), round(s, 3)) for s, p in scored[:limit]]


def search_all(searcher, claims: list[dict], per_query: int, queries_per_claim: int = 1) -> tuple[dict[str, dict], list[str]]:
    """Run all queries (parallel). Returns ({normalized_url: result}, error messages)."""
    jobs = [(c["id"], q) for c in claims for q in c["queries"][:max(1, queries_per_claim)]]
    errors: list[str] = []
    results: dict[str, dict] = {}

    def one(job):
        cid, q = job
        try:
            return cid, q, searcher.search(q, per_query), None
        except SearchError as exc:
            return cid, q, [], exc

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(jobs)))) as ex:
        outcomes = list(ex.map(one, jobs))
    fatal = None
    for cid, q, res, err in outcomes:
        if err:
            errors.append(f"search '{q[:60]}': {err.kind}")
            fatal = fatal or err
            continue
        for rank, r in enumerate(res):
            key = normalize_url(r["url"])
            entry = results.setdefault(key, {**r, "claim_ids": set(), "best_rank": rank, "queries": set()})
            entry["claim_ids"].add(cid)
            entry["queries"].add(q)
            entry["best_rank"] = min(entry["best_rank"], rank)
            if len(r["snippets"]) > len(entry["snippets"]):
                entry["snippets"] = r["snippets"]
    if not results and fatal:
        raise fatal  # every query failed -> surface as search failure
    return results, errors


def rank_results(results: dict[str, dict], claims: list[dict], limit: int) -> list[dict]:
    claim_tok = {c["id"]: tokens(c["text"]) for c in claims}
    ranked = []
    for r in results.values():
        q = classify_source(r["url"])
        rel = max((overlap(claim_tok[cid], r["title"] + " " + " ".join(r["snippets"])) for cid in r["claim_ids"]), default=0)
        score = 0.45 * TIER_WEIGHT[q["tier"]] + 0.45 * rel + 0.10 * (1 / (1 + r["best_rank"]))
        ranked.append({**r, "quality": q, "snippet_relevance": round(rel, 3), "score": score})
    ranked.sort(key=lambda r: -r["score"])
    # ensure every claim gets coverage: round-robin over claims, then fill by score
    chosen, seen = [], set()
    for cid in [c["id"] for c in claims]:
        for r in ranked:
            if cid in r["claim_ids"] and r["url"] not in seen:
                chosen.append(r); seen.add(r["url"]); break
    for r in ranked:
        if len(chosen) >= limit:
            break
        if r["url"] not in seen:
            chosen.append(r); seen.add(r["url"])
    return chosen[:limit]


def build_evidence(chosen: list[dict], claims: list[dict], fetcher: SafeFetcher) -> tuple[list[dict], list[str]]:
    claim_texts = {c["id"]: c["text"] for c in claims}

    def work(r):
        try:
            page = fetcher.fetch_page(r["url"])
            return r, page, None
        except FetchError as exc:
            return r, None, exc.reason

    with ThreadPoolExecutor(max_workers=min(6, max(1, len(chosen)))) as ex:
        fetched = list(ex.map(work, chosen))

    evidence, notes, seen_passages = [], [], set()
    for r, page, err in fetched:
        texts = [claim_texts[c] for c in sorted(r["claim_ids"])]
        passages: list[dict] = []
        published = r.get("age")
        title = r["title"]
        if page is not None:
            title = page.title or title
            published = page.published or published
            for text, score in select_passages(page.paragraphs, texts):
                passages.append({"text": text, "origin": "page", "relevance": score})
        else:
            notes.append(f"{r['quality']['domain']}: page not retrieved ({err}); using search snippets only")
        if not passages:  # fall back to the excerpts returned by the search step
            for text, score in select_passages(r["snippets"], texts):
                passages.append({"text": text, "origin": "search_snippet", "relevance": score})
        # drop duplicate passages already used by another source
        unique = []
        for p in passages:
            h = hashlib.sha1(re.sub(r"\W+", "", p["text"].lower()).encode()).hexdigest()
            if h not in seen_passages:
                seen_passages.add(h)
                unique.append(p)
        if not unique:
            continue
        evidence.append({
            "url": r["url"], "title": trim(title, 160), "domain": r["quality"]["domain"],
            "quality": {"tier": r["quality"]["tier"], "meaning": r["quality"]["meaning"]},
            "published": published, "passages": unique,
            "relevance": round(max(p["relevance"] for p in unique), 3),
            "retrieval": {"page_fetched": page is not None, "error": err},
            "claim_ids": sorted(r["claim_ids"]),
        })
    evidence.sort(key=lambda e: -(TIER_WEIGHT[e["quality"]["tier"]] * 0.5 + e["relevance"] * 0.5))
    for i, e in enumerate(evidence, 1):
        e["id"] = f"E{i}"
    return evidence, notes