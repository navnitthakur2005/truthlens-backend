import json, requests
from app.config import get_config
from app.retrieval.groq_search import GroqSearchClient, SearchError
from app.retrieval import evidence as ev
from app.retrieval.claims import extract_claims
from app.retrieval.fetcher import SafeFetcher
from app.verification.llm import GroqClient

cfg = get_config()
QUERY = "ISRO Chandrayaan-3 Moon landing date"

print("\n=== STEP 1: raw Groq call (max_tokens=120, as in your code) ===")
for mt in (120, 500):
    payload = {
        "model": cfg.groq_search_model,
        "messages": [
            {"role": "system", "content": "You are a search assistant. Always use web search. Reply in one short sentence."},
            {"role": "user", "content": f"Search the web for reliable sources about: {QUERY}"},
        ],
        "temperature": 0, "max_tokens": mt,
    }
    r = requests.post(cfg.groq_api_base, json=payload, timeout=60,
                      headers={"Authorization": f"Bearer {cfg.groq_api_key}"})
    print(f"\nmax_tokens={mt} -> HTTP {r.status_code}")
    if r.status_code != 200:
        print("ERROR BODY:", r.text[:400]); continue
    msg = r.json()["choices"][0]["message"]
    tools = msg.get("executed_tools") or []
    print("message keys:", list(msg.keys()))
    print("executed_tools count:", len(tools))
    n = sum(len((t.get("search_results") or {}).get("results") or []) for t in tools)
    print("search results found:", n)

print("\n=== STEP 2: your GroqSearchClient ===")
s = GroqSearchClient(cfg.groq_api_key, cfg.groq_search_model, cfg.groq_api_base)
try:
    res = s.search(QUERY, 6)
    print("results:", len(res))
    for x in res[:3]: print(" -", x["url"], "| snippets:", len(x["snippets"]))
except SearchError as e:
    print("SearchError:", e.kind, e)

print("\n=== STEP 3: full retrieval pipeline ===")
llm = GroqClient(cfg.groq_api_key, cfg.groq_model, cfg.groq_api_base)
claims, warn = extract_claims(llm, "India's Chandrayaan-3 landed near the Moon's south pole in August 2023", "", 3)
print("claims:", [(c["id"], c["source"], c["queries"]) for c in claims], "| warn:", warn)
try:
    results, errs = ev.search_all(s, claims, cfg.results_per_query, cfg.queries_per_claim)
    print("search results:", len(results), "| errors:", errs)
    chosen = ev.rank_results(results, claims, cfg.max_pages_to_fetch)
    print("chosen:", len(chosen))
    fetcher = SafeFetcher(cfg.http_timeout, cfg.max_page_bytes, cfg.user_agent, cfg.respect_robots)
    evidence, notes = ev.build_evidence(chosen, claims, fetcher)
    print("evidence items kept:", len(evidence))
    for n in notes: print(" note:", n)
except SearchError as e:
    print("PIPELINE SEARCH FAILED:", e.kind, e)