"""Flask API. Artifacts load at startup. Serving does not need MongoDB.

    python -m pl2016.app
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pl2016 import __version__
from pl2016.ask import Models, UnsupportedState, ask
from pl2016.config import STATES
from pl2016.train import ARTIFACTS


STATIC = Path(__file__).resolve().parent / "static"


def create_app(models: Models | None = None, artifacts: Path | None = None):
    from flask import Flask, jsonify, request, send_from_directory

    directory = artifacts or Path(os.environ.get("PL2016_ARTIFACTS", ARTIFACTS))
    loaded = models if models is not None else Models.load(directory)
    app = Flask(__name__)
    app.config["MODELS"] = loaded
    app.config["ARTIFACTS"] = directory

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/favicon.ico")
    def favicon():
        return ("", 204)

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "version": __version__})

    @app.get("/manifest")
    def manifest():
        path = directory / "manifest.json"
        if path.exists():
            return jsonify(json.loads(path.read_text(encoding="utf-8")))
        return jsonify({"version": __version__, "states": list(STATES)})

    @app.post("/ask")
    def ask_route():
        payload = request.get_json(silent=True) or {}
        question = payload.get("question") or ""
        state = payload.get("state") or ""
        if not str(question).strip() or not str(state).strip():
            return jsonify({"error": "question and state are required"}), 400
        try:
            return jsonify(ask(str(question), str(state), loaded))
        except UnsupportedState as exc:
            return jsonify({"error": str(exc)}), 400
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    return app


def main() -> None:
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))


if __name__ == "__main__":
    main()
