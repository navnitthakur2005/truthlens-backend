"""DEV-ONLY stand-ins for Brave Search and Groq, used to exercise the app end-to-end offline.

NOT part of the product: results are canned and say so. Run:  python -m tests.e2e.mock_services
Then start the backend with BRAVE_API_BASE / GROQ_API_BASE pointing here (see README, 'Offline demo').
"""
import json
import re
import threading

from flask import Flask, jsonify, request

brave, groq = Flask("mock-brave"), Flask("mock-groq")


@brave.get("/search")
def search():
    q = request.args.get("q", "")
    return jsonify({"web": {"results": [
        {"title": "Chandrayaan-3 Mission: Soft-landing - ISRO (MOCK RESULT)",
         "url": "https://www.isro.gov.in/Chandrayaan3.html",
         "description": "Chandrayaan-3 Vikram lander soft-landed on the Moon on 23 August 2023, near the lunar south pole.",
         "extra_snippets": ["India became the fourth country to achieve a soft landing on the Moon after the Chandrayaan-3 landing."],
         "page_age": "2023-08-24T00:00:00"},
        {"title": "India lands Chandrayaan-3 on the Moon - Reuters (MOCK RESULT)",
         "url": "https://www.reuters.com/science/india-moon-landing-2023-08-23/",
         "description": "India's Chandrayaan-3 spacecraft landed on the Moon on Wednesday 23 August 2023, a historic first near the south pole.",
         "page_age": "2023-08-23T00:00:00"},
        {"title": "My blog: Chandrayaan-3 landed on the Moon in 2024 (MOCK RESULT)",
         "url": "https://blog.example.net/chandrayaan-2024",
         "description": "I think Chandrayaan-3 landed on the Moon in 2024, trust me.", "page_age": None},
    ]}})


@groq.post("/v1/chat/completions")
def chat():
    body = request.get_json()
    system, user = body["messages"][0]["content"], body["messages"][1]["content"]
    if "extract checkable factual claims" in system:
        h = re.search(r"HEADLINE: (.*)", user)
        claim = (h.group(1) if h else "Chandrayaan-3 landed on the Moon in 2024.").strip()
        out = {"claims": [{"text": claim, "queries": ["Chandrayaan-3 Moon landing date"]}]}
    else:
        ev = json.loads(re.search(r"<untrusted_evidence>\n(.*)\n</untrusted_evidence>", user, re.S).group(1))
        a = [{"evidence_id": e["id"], "relation": "contradicts" if e["source_tier"] in ("official", "news") else "inconclusive",
              "rationale": "States 23 August 2023, not 2024." if e["source_tier"] in ("official", "news") else "Unsourced assertion."}
             for e in ev]
        out = {"overall_verdict": "contradicted",
               "summary": "(Mock verifier) Official and news sources give 23 August 2023 as the landing date, so the year 2024 is wrong.",
               "claims": [{"claim_id": "C1", "verdict": "contradicted", "explanation": "The year in the claim conflicts with the sources.",
                           "correction": "The landing happened on 23 August 2023.", "assessments": a}],
               "limitations": ["Mock services were used: this run does not reflect real web evidence."]}
    return jsonify({"choices": [{"message": {"content": json.dumps(out)}}]})


if __name__ == "__main__":
    t = threading.Thread(target=lambda: brave.run(port=5101), daemon=True)
    t.start()
    groq.run(port=5102)
