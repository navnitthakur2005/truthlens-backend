"""Train and compare Naive Bayes / Logistic Regression / Linear SVM for each input mode.

    python -m training.train [--data-dir data/processed] [--artifact-dir artifacts]

For every mode (headline, article, both) and every algorithm a pipeline
``TfidfVectorizer -> classifier`` is fitted on the TRAIN split only. Models are compared on the
VALIDATION split (macro-F1); the TEST split is reported but never used for selection.
LinearSVC is wrapped in CalibratedClassifierCV (sigmoid) so its probabilities are calibrated.

Writes  artifacts/<mode>.joblib  and  artifacts/metrics.json.  Nothing is written unless training
actually completes.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from app.ml.cleaning import build_text

BACKEND = Path(__file__).resolve().parent.parent
SEED = 42
MODES = ("headline", "article", "both")

TFIDF_PARAMS = dict(
    lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=3, max_df=0.9,
    max_features=100_000, sublinear_tf=True, stop_words="english",
)


def make_models() -> dict:
    return {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(C=4.0, max_iter=1000, random_state=SEED),
        "Linear SVM (calibrated)": CalibratedClassifierCV(
            LinearSVC(C=0.5, random_state=SEED), method="sigmoid", cv=3
        ),
    }


def make_pipeline(clf) -> Pipeline:
    return Pipeline([("tfidf", TfidfVectorizer(**TFIDF_PARAMS)), ("clf", clf)])


def evaluate(pipe: Pipeline, X, y) -> dict:
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_fake": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "f1_fake": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "precision_real": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "recall_real": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "brier_score": float(brier_score_loss(y, proba)),
        "confusion_matrix": {"labels": ["real", "fake"], "matrix": cm.tolist()},
    }


def load_split(data_dir: Path, name: str, mode: str):
    df = pd.read_csv(data_dir / f"{name}.csv", keep_default_na=False)
    X = [build_text(t, x, mode) for t, x in zip(df["title"], df["text"])]
    y = df["label"].to_numpy()
    keep = np.array([len(s) > 0 for s in X])
    return [s for s, k in zip(X, keep) if k], y[keep]


def train_all(data_dir: Path, artifact_dir: Path, smoke_test: bool = False) -> dict:
    report_path = data_dir / "cleaning_report.json"
    if not (data_dir / "train.csv").exists():
        raise FileNotFoundError(f"{data_dir}/train.csv missing - run `python -m training.prepare_data` first")
    cleaning = json.loads(report_path.read_text()) if report_path.exists() else {}

    artifact_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    started = time.time()
    staged: dict = {}

    for mode in MODES:
        Xtr, ytr = load_split(data_dir, "train", mode)
        Xva, yva = load_split(data_dir, "val", mode)
        Xte, yte = load_split(data_dir, "test", mode)
        mode_res = {"sizes": {"train": len(ytr), "val": len(yva), "test": len(yte)}, "models": {}}
        fitted = {}
        for name, clf in make_models().items():
            t0 = time.time()
            pipe = make_pipeline(clf).fit(Xtr, ytr)  # TF-IDF vocabulary learned on TRAIN only
            fit_s = time.time() - t0
            mode_res["models"][name] = {
                "fit_seconds": round(fit_s, 2),
                "validation": evaluate(pipe, Xva, yva),
                "test": evaluate(pipe, Xte, yte),
            }
            fitted[name] = pipe
            m = mode_res["models"][name]
            print(f"[{mode:8s}] {name:26s} val macro-F1={m['validation']['macro_f1']:.4f} "
                  f"test macro-F1={m['test']['macro_f1']:.4f} ({fit_s:.0f}s)")
        best = max(mode_res["models"], key=lambda n: mode_res["models"][n]["validation"]["macro_f1"])
        mode_res["selected_model"] = best
        mode_res["selection_rule"] = "highest validation macro-F1 (test split not used for selection)"
        results[mode] = mode_res
        staged[mode] = {
            "pipeline": fitted[best],
            "model_name": best,
            "mode": mode,
            "probabilities_calibrated": True,
            "sklearn_version": sklearn.__version__,
            "smoke_test": smoke_test,
        }

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "smoke_test": smoke_test,
        "dataset": {
            "name": "SYNTHETIC SMOKE-TEST DATA (not WELFake)" if smoke_test else "WELFake",
            "cleaning": cleaning,
            "label_mapping": {"0": "real", "1": "fake"},
        },
        "tfidf_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in TFIDF_PARAMS.items()},
        "seed": SEED,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__},
        "total_training_seconds": round(time.time() - started, 1),
        "modes": results,
    }
    # write only once everything succeeded
    for mode, art in staged.items():
        joblib.dump(art, artifact_dir / f"{mode}.joblib", compress=3)
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=BACKEND / "data" / "processed")
    ap.add_argument("--artifact-dir", type=Path, default=BACKEND / "artifacts")
    ap.add_argument("--smoke-test", action="store_true", help="Mark artifacts as synthetic smoke-test output")
    args = ap.parse_args(argv)
    try:
        train_all(args.data_dir, args.artifact_dir, args.smoke_test)
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}")
        return 2
    print(f"Saved models + metrics to {args.artifact_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
