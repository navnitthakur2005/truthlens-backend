import json

import pytest

from app.config import Config
from app.history.store import HistoryStore
from app.ml.predictor import Predictor
from app.retrieval.fetcher import FetchError, Page
from app.services import AnalysisService
from training import prepare_data, train
from training.make_smoke_data import make


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory):
    """Trains on SYNTHETIC data once per test session (tests code paths, not model quality)."""
    root = tmp_path_factory.mktemp("trained")
    csv = root / "raw.csv"
    make(n_per_class=150).to_csv(csv, index=False)
    prepare_data.run(csv, root / "processed", source_fake_value=1)
    train.train_all(root / "processed", root / "artifacts", smoke_test=True)
    return root


# ---- fakes for external services -------------------------------------------------
class FakeBrave:
    configured = True

    def __init__(self, results=None, error=None):
        self.results, self.error, self.queries = results, error, []

    def search(self, q, count=6):
        self.queries.append(q)
        if self.error:
            raise self.error
        return self.results if self.results is not None else [
            {"title": "Chandrayaan-3 Mission - ISRO", "url": "https://www.isro.gov.in/Chandrayaan3.html",
             "snippets": ["Chandrayaan-3 soft-landed on the Moon on 23 August 2023, making India the fourth country to do so."],
             "age": "2023-08-24"},
            {"title": "Random blog", "url": "https://blog.example.net/post",
             "snippets": ["Chandrayaan-3 landed on the Moon in 2024 says our blog."], "age": None},
        ]


class FakeLLM:
    def __init__(self, responses=None, configured=True, error=None):
        self.responses, self.configured, self.error, self.calls = list(responses or []), configured, error, []

    def chat_json(self, system, user, **kw):
        self.calls.append((system, user))
        if self.error:
            raise self.error
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self.responses.pop(0)


class FakeFetcher:
    def __init__(self, pages=None):
        self.pages = pages or {}

    def fetch_page(self, url):
        if url in self.pages:
            return self.pages[url]
        raise FetchError("access_denied", "403")


CLAIMS_JSON = {"claims": [{"text": "Chandrayaan-3 landed on the Moon in 2024.",
                           "queries": ["Chandrayaan-3 Moon landing date"]}]}
VERDICT_JSON = {
    "overall_verdict": "contradicted",
    "summary": "Official evidence places the landing in August 2023, not 2024.",
    "claims": [{"claim_id": "C1", "verdict": "contradicted", "explanation": "The landing year is wrong.",
                "correction": "The landing happened on 23 August 2023.",
                "assessments": [{"evidence_id": "E1", "relation": "contradicts", "rationale": "Gives 2023."}]}],
    "limitations": ["Only one official source was found."],
}


def make_service(tmp_path, artifact_dir, brave=None, llm=None, fetcher=None):
    cfg = Config(artifact_dir=artifact_dir, history_db=tmp_path / "h.sqlite3")
    return AnalysisService(cfg, Predictor(artifact_dir), HistoryStore(cfg.history_db),
                           brave=brave or FakeBrave(), llm=llm or FakeLLM([CLAIMS_JSON, VERDICT_JSON] * 10),
                           fetcher=fetcher or FakeFetcher())


@pytest.fixture
def svc_factory(tmp_path, trained_dir):
    def f(**kw):
        return make_service(tmp_path, kw.pop("artifact_dir", trained_dir / "artifacts"), **kw)
    return f


@pytest.fixture
def client_factory(svc_factory):
    from app import create_app

    def f(**kw):
        svc = svc_factory(**kw)
        app = create_app(svc.cfg, svc)
        return app.test_client(), svc
    return f
