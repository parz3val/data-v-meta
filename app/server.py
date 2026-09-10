"""Flask front end for the trained models: python -m flask --app app.server run --port 5055
Set WQA_MODE (default full) and WQA_ENCODER=0 to skip loading the text encoder."""
from __future__ import annotations
import os
from flask import Flask, jsonify, render_template, request
from app.scoring import Scorer, GRADES

app = Flask(__name__, template_folder="templates")
scorer = Scorer(mode=os.environ.get("WQA_MODE", "full"), encoder=os.environ.get("WQA_ENCODER", "1") != "0")


@app.get("/")
def index():
    return render_template("index.html", grades=GRADES, has_encoder=scorer.enc is not None)


@app.get("/api/grade")
def grade():
    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"error": "give a title or a Wikipedia URL"}), 400
    try:
        return jsonify(scorer.score(title))
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"scoring failed: {e}"}), 500


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5055")))
