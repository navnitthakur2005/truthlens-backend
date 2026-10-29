from unittest.mock import MagicMock

import pytest

from app import create_app
from app.config import Config
from app.services import FactCheckService, _extract_json


def fake_response(content, results=None, status=200):
    r = MagicMock()
    r.status_code = status
    r.headers = {}
    r.json.return_value = {"choices": [{"message": {
        "content": content,
        "executed_tools": [{"type": "search", "search_results": {"results": results or []}}],
    }}]}
    return r


def make_client(resp):
    session = MagicMock()
    session.post.return_value = resp
    cfg = Config(groq_api_key="test-key")
    return create_app(cfg, FactCheckService(cfg, session)).test_client(), session


GOOD = ('Here you go:\n```json\n{"verdict":"false","confidence":92,"summary":"It did not happen.",'
        '"claims":[{"claim":"X","verdict":"false","explanation":"No."}],'
        '"sources":[{"title":"Reuters","url":"https://reuters.com/a"},{"title":"Fake","url":"https://made-up.example"}]}\n```')


def test_happy_path_filters_invented_sources():
    c, session = make_client(fake_response(GOOD, [{"title": "Reuters", "url": "https://reuters.com/a"}]))
    r = c.post("/api/analyze", json={"text": "Some news claim to check please"})
    assert r.status_code == 200
    d = r.get_json()
    assert d["verdict"] == "false" and d["confidence"] == 92 and d["searched"] is True
    assert [s["url"] for s in d["sources"]] == ["https://reuters.com/a"]
    assert session.post.call_args.kwargs["json"]["tools"] == [{"type": "browser_search"}]


def test_unparseable_reply_is_unverified_not_crash():
    c, _ = make_client(fake_response("I could not format this."))
    d = c.post("/api/analyze", json={"text": "Some news claim to check please"}).get_json()
    assert d["verdict"] == "unverified" and d["searched"] is False


def test_validation_and_missing_key():
    c, _ = make_client(fake_response(GOOD))
    assert c.post("/api/analyze", json={"text": "short"}).status_code == 400
    assert c.post("/api/analyze", data="nope").status_code == 400
    cfg = Config(groq_api_key="")
    c2 = create_app(cfg, FactCheckService(cfg)).test_client()
    assert c2.post("/api/analyze", json={"text": "Some news claim to check please"}).status_code == 503


def test_groq_errors_become_messages():
    bad = MagicMock(status_code=401, headers={})
    c, _ = make_client(bad)
    r = c.post("/api/analyze", json={"text": "Some news claim to check please"})
    assert r.status_code == 502 and "API key" in r.get_json()["error"]


def test_compound_model_sends_no_tools():
    session = MagicMock()
    session.post.return_value = fake_response(GOOD)
    cfg = Config(groq_api_key="k", groq_model="groq/compound")
    create_app(cfg, FactCheckService(cfg, session)).test_client().post("/api/analyze", json={"text": "Some news claim to check please"})
    assert "tools" not in session.post.call_args.kwargs["json"]


def test_extract_json_handles_braces_in_strings():
    assert _extract_json('x {"a": "}{", "b": 1} y') == {"a": "}{", "b": 1}

def test_falls_back_to_next_model_on_rate_limit_and_413():
    limited = MagicMock(status_code=429, headers={})
    limited.json.return_value = {"error": {"message": "Rate limit reached for model"}}
    toobig = MagicMock(status_code=413, headers={})
    toobig.json.return_value = {"error": {"message": "Request too large"}}
    session = MagicMock()
    session.post.side_effect = [limited, toobig, fake_response(GOOD, [{"title": "R", "url": "https://reuters.com/a"}])]
    cfg = Config(groq_api_key="k", groq_model="openai/gpt-oss-120b",
                 groq_fallback_models=["openai/gpt-oss-20b", "groq/compound-mini"])
    c = create_app(cfg, FactCheckService(cfg, session)).test_client()
    d = c.post("/api/analyze", json={"text": "Some news claim to check please"}).get_json()
    assert d["verdict"] == "false" and d["model"] == "groq/compound-mini"
    assert "tool_choice" not in session.post.call_args_list[0].kwargs["json"]


def test_bad_key_does_not_try_other_models_and_shows_groq_message():
    bad = MagicMock(status_code=401, headers={})
    bad.json.return_value = {"error": {"message": "Invalid API Key"}}
    session = MagicMock()
    session.post.return_value = bad
    cfg = Config(groq_api_key="k")
    r = create_app(cfg, FactCheckService(cfg, session)).test_client().post(
        "/api/analyze", json={"text": "Some news claim to check please"})
    assert r.status_code == 502 and "Invalid API Key" in r.get_json()["error"]
    assert session.post.call_count == 1
