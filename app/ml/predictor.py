"""Loads the saved per-mode pipelines and produces an ML prediction (NOT a factual verdict)."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import joblib

from app.config import MODES
from app.ml.cleaning import build_text

DISCLAIMER = (
    "ML prediction, not a factual verdict: it estimates whether the wording resembles articles labelled "
    "fake or real in the training dataset."
)


class ModelUnavailable(RuntimeError):
    pass


class Predictor:
    def __init__(self, artifact_dir: Path):
        self.dir = Path(artifact_dir)
        self._models: dict = {}
        self._lock = threading.Lock()

    def metrics(self) -> dict:
        p = self.dir / "metrics.json"
        if not p.exists():
            raise ModelUnavailable(
                "No trained model metrics found. Run: python -m training.prepare_data && python -m training.train"
            )
        return json.loads(p.read_text())

    def available_modes(self) -> list[str]:
        return [m for m in MODES if (self.dir / f"{m}.joblib").exists()]

    def _load(self, mode: str) -> dict:
        with self._lock:
            if mode not in self._models:
                path = self.dir / f"{mode}.joblib"
                if not path.exists():
                    raise ModelUnavailable(
                        f"Model for mode '{mode}' not found at {path}. "
                        "Train it with: python -m training.prepare_data && python -m training.train"
                    )
                try:
                    art = joblib.load(path)
                except Exception as exc:  # corrupt / version mismatch
                    raise ModelUnavailable(f"Could not load model '{mode}': {exc}") from exc
                if not isinstance(art, dict) or "pipeline" not in art:
                    raise ModelUnavailable(f"Model file {path} has an unexpected format")
                self._models[mode] = art
            return self._models[mode]

    def predict(self, mode: str, title: str = "", text: str = "") -> dict:
        if mode not in MODES:
            raise ValueError(f"Unknown mode '{mode}'")
        art = self._load(mode)
        x = build_text(title, text, mode)
        if not x:
            raise ValueError("Nothing to classify for this input mode")
        pipe = art["pipeline"]
        proba = pipe.predict_proba([x])[0]
        classes = list(pipe.classes_)
        p_fake = float(proba[classes.index(1)])
        label = "fake" if p_fake >= 0.5 else "real"
        return {
            "label": label,
            "probability": round(p_fake if label == "fake" else 1 - p_fake, 4),
            "probabilities": {"fake": round(p_fake, 4), "real": round(1 - p_fake, 4)},
            "model_name": art["model_name"],
            "mode": mode,
            "probabilities_calibrated": bool(art.get("probabilities_calibrated", False)),
            "smoke_test_model": bool(art.get("smoke_test", False)),
            "disclaimer": DISCLAIMER,
        }
