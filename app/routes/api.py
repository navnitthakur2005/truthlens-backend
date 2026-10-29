"""REST API: GET /api/health, POST /api/analyze."""
from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from app.services import FactCheckError, ValidationError

bp = Blueprint("api", __name__, url_prefix="/api")


def _svc():
    return current_app.extensions["factcheck"]


@bp.get("/health")
def health():
    s = _svc()
    return jsonify({
        "status": "ok",
        "groq_configured": s.configured,
        "model": s.cfg.groq_model,
        "limits": {"min": s.cfg.min_chars, "max": s.cfg.max_chars},
    })


@bp.post("/analyze")
def analyze():
    s = _svc()
    data = request.get_json(silent=True)
    try:
        text = s.validate(data)
        return jsonify(s.check(text))
    except ValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    except FactCheckError as exc:
        return jsonify({"error": str(exc)}), exc.status