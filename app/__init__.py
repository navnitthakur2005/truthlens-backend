"""Fake-news checker backend (Flask + Groq web search)."""
from __future__ import annotations

import logging

from flask import Flask, jsonify
from flask_cors import CORS

from app.config import Config, get_config
from app.routes.api import bp as api_bp
from app.services import FactCheckService


def create_app(cfg: Config | None = None, service: FactCheckService | None = None) -> Flask:
    cfg = cfg or get_config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    app.extensions["factcheck"] = service or FactCheckService(cfg)
    app.register_blueprint(api_bp)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"error": "Request body is too large."}), 413

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "Not found."}), 404

    @app.errorhandler(405)
    def bad_method(_):
        return jsonify({"error": "Method not allowed."}), 405

    @app.errorhandler(500)
    def server_error(_):
        return jsonify({"error": "Internal server error."}), 500

    return app
