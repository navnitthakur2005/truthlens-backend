import os
import json
import logging
from groq import Groq

logger = logging.getLogger(__name__)

class LLMError(Exception):
    pass

class GroqClient:
    def __init__(self, api_key: str = None, model: str = None, api_base: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        
        if self.api_key:
            kwargs = {"api_key": self.api_key}
            # Clean up base URL if accidentally duplicated
            if api_base:
                base = api_base.split("/openai/v1")[0]
                kwargs["base_url"] = base
            self.client = Groq(**kwargs)
        else:
            self.client = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1, **kwargs) -> dict:
        if not self.configured or not self.client:
            raise LLMError("GROQ_API_KEY is missing or client is not configured.")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }
        
        if "max_tokens" in kwargs and kwargs["max_tokens"]:
            payload["max_tokens"] = kwargs["max_tokens"]

        try:
            response = self.client.chat.completions.create(**payload)
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            if "rate_limit" in str(e).lower():
                raise LLMError("rate_limited: Groq rate limit reached")
            raise LLMError(str(e))

    def verify_claim(self, text: str, search_evidence: str = "") -> dict:
        prompt = f"""You are a strict fact-checker. Analyze the following news claim and web search evidence to determine if the news is correct, false, or misleading.

News / Claim:
"{text}"

Web Search Evidence:
{search_evidence if search_evidence else "No external search evidence available."}

Respond ONLY with a valid JSON object in this exact format (no markdown formatting outside JSON):
{{
    "verdict": "correct" (or "false" or "misleading" or "unverified"),
    "confidence": 85,
    "summary": "A concise 2-3 sentence explanation explaining why this claim is correct or false based on the evidence."
}}"""
        return self.chat_json("You are a precise fact-checking assistant that outputs strict JSON.", prompt)

def verify_news_with_groq(text: str, search_evidence: str = "") -> dict:
    try:
        client = GroqClient()
        return client.verify_claim(text, search_evidence)
    except Exception as e:
        return {
            "verdict": "unverified",
            "confidence": 0,
            "summary": f"Verification error: {str(e)}"
        }