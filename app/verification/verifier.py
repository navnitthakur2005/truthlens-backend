import logging

logger = logging.getLogger(__name__)

VERDICT_LABELS = {
    "correct": "Correct",
    "false": "False",
    "misleading": "Misleading",
    "unverified": "Misleading"
}

# Map everything safely to frontend-supported style keys
SUPPORTED_UI_VERDICTS = {"correct", "false", "misleading"}

class ClaimResult:
    def __init__(self, data: dict, idx: int = 0):
        self.claim_id = data.get("claim_id", f"claim_{idx}")
        self.text = data.get("text", "")
        
        v = str(data.get("verdict", "misleading")).lower().strip()
        self.verdict = v if v in SUPPORTED_UI_VERDICTS else "misleading"
        
        self.confidence = data.get("confidence", 50)
        self.summary = data.get("summary", "")

    def model_dump(self):
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "summary": self.summary
        }

class VerificationResult:
    def __init__(self, data: dict):
        v = str(data.get("verdict", "misleading")).lower().strip()
        norm_verdict = v if v in SUPPORTED_UI_VERDICTS else "misleading"

        self.verdict = norm_verdict
        self.overall_verdict = norm_verdict
        self.confidence = data.get("confidence", 50)
        self.summary = data.get("summary", "")
        
        raw_claims = data.get("claims", [])
        self.claims = [
            ClaimResult(c, i) if isinstance(c, dict) else c 
            for i, c in enumerate(raw_claims)
        ]
        self.evidence = data.get("evidence", [])
        self.limitations = []

    def get(self, key, default=None):
        if key == "claims":
            return [c.model_dump() if hasattr(c, "model_dump") else c for c in self.claims]
        return getattr(self, key, default)

    def to_dict(self):
        return {
            "verdict": self.verdict,
            "overall_verdict": self.overall_verdict,
            "confidence": self.confidence,
            "summary": self.summary,
            "claims": [c.model_dump() if hasattr(c, "model_dump") else c for c in self.claims],
            "evidence": self.evidence
        }

def verify(llm, claims: list, evidence: list = None) -> tuple:
    logger.info("Running verification on extracted claims...")
    
    evidence_text = ""
    if evidence:
        evidence_text = "\n".join([e.get("content", "") for e in evidence if isinstance(e, dict)])
    
    assessed_claims = []
    overall_verdict = "misleading"
    max_confidence = 0
    summaries = []

    if not claims:
        claims = [{"text": "General news claim analysis"}]

    for i, claim in enumerate(claims):
        claim_text = claim.get("text", "") if isinstance(claim, dict) else str(claim)
        claim_id = claim.get("claim_id", f"claim_{i}")
        
        try:
            if hasattr(llm, "verify_claim"):
                res = llm.verify_claim(claim_text, search_evidence=evidence_text)
            else:
                prompt = f"Analyze this claim based on evidence: {claim_text}\nEvidence: {evidence_text}\nReturn JSON with verdict (correct/false/misleading), confidence (0-100), summary."
                res = llm.chat_json("You are a fact checker.", prompt)
        except Exception as e:
            logger.error(f"Verification call failed: {e}")
            res = {"verdict": "misleading", "confidence": 0, "summary": str(e)}

        raw_v = str(res.get("verdict", "misleading")).lower().strip()
        verdict = raw_v if raw_v in SUPPORTED_UI_VERDICTS else "misleading"
        
        confidence = res.get("confidence", 50)
        summary = res.get("summary", "")

        assessed_claims.append({
            "claim_id": claim_id,
            "text": claim_text,
            "verdict": verdict,
            "confidence": confidence,
            "summary": summary
        })

        if confidence >= max_confidence:
            max_confidence = confidence
            overall_verdict = verdict

        if summary:
            summaries.append(summary)

    final_summary = " ".join(summaries) if summaries else "Analyzed via Groq."

    result_data = {
        "verdict": overall_verdict,
        "confidence": max_confidence,
        "summary": final_summary,
        "claims": assessed_claims,
        "evidence": evidence or []
    }

    return VerificationResult(result_data), []