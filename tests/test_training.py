import json

import numpy as np
import pandas as pd
import pytest

from app.ml.predictor import ModelUnavailable, Predictor
from training.prepare_data import assert_no_leakage
from training.train import evaluate


def test_splits_are_stratified_and_leak_free(trained_dir):
    d = trained_dir / "processed"
    parts = {n: pd.read_csv(d / f"{n}.csv", keep_default_na=False) for n in ("train", "val", "test")}
    assert_no_leakage(parts["train"], parts["val"], parts["test"])
    ratios = [p["label"].mean() for p in parts.values()]
    assert max(ratios) - min(ratios) < 0.05
    rep = json.loads((d / "cleaning_report.json").read_text())
    assert rep["rows_after"] == sum(len(p) for p in parts.values())


def test_metrics_file_structure(trained_dir):
    m = json.loads((trained_dir / "artifacts" / "metrics.json").read_text())
    assert m["smoke_test"] is True
    for mode in ("headline", "article", "both"):
        mm = m["modes"][mode]
        assert set(mm["models"]) == {"Multinomial Naive Bayes", "Logistic Regression", "Linear SVM (calibrated)"}
        assert mm["selected_model"] in mm["models"]
        for res in mm["models"].values():
            for split in ("validation", "test"):
                e = res[split]
                assert {"accuracy", "precision_macro", "recall_macro", "macro_f1", "confusion_matrix"} <= set(e)
                assert sum(map(sum, e["confusion_matrix"]["matrix"])) == e["n"]


def test_evaluate_metrics_known_values():
    class P:  # perfect-then-wrong predictor
        def predict(self, X): return np.array([0, 0, 1, 1])
        def predict_proba(self, X): return np.array([[.9, .1], [.8, .2], [.3, .7], [.4, .6]])
    e = evaluate(P(), ["a"] * 4, np.array([0, 1, 1, 1]))
    assert e["accuracy"] == pytest.approx(0.75)
    assert e["confusion_matrix"]["matrix"] == [[1, 0], [1, 2]]
    assert e["precision_fake"] == pytest.approx(1.0) and e["recall_fake"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("mode", ["headline", "article", "both"])
def test_predictor_all_modes(trained_dir, mode):
    p = Predictor(trained_dir / "artifacts")
    r = p.predict(mode, "SHOCKING: they are hiding the truth about energy", "Share this before they delete it! Wake up people.")
    assert r["label"] in ("fake", "real") and 0.5 <= r["probability"] <= 1
    assert r["mode"] == mode and r["probabilities_calibrated"] is True
    assert r["probabilities"]["fake"] + r["probabilities"]["real"] == pytest.approx(1, abs=1e-3)
    assert "not a factual verdict" in r["disclaimer"]


def test_predictor_missing_artifacts(tmp_path):
    p = Predictor(tmp_path)
    with pytest.raises(ModelUnavailable):
        p.predict("headline", "Some headline text here", "")
    with pytest.raises(ModelUnavailable):
        p.metrics()
    assert p.available_modes() == []


def test_predictor_corrupt_artifact(tmp_path):
    (tmp_path / "headline.joblib").write_bytes(b"not a model")
    with pytest.raises(ModelUnavailable):
        Predictor(tmp_path).predict("headline", "Some headline text here", "")


def test_predictor_rejects_unknown_mode_and_empty(trained_dir):
    p = Predictor(trained_dir / "artifacts")
    with pytest.raises(ValueError):
        p.predict("bogus", "x", "y")
    with pytest.raises(ValueError):
        p.predict("headline", "", "body only")
