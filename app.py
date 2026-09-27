"""QuestGen AI — Flask web app for NLP-based question generation.

    python app.py
Then open http://127.0.0.1:5000 in a browser.

First-time setup:
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
"""

import os

from flask import Flask, jsonify, render_template, request, send_from_directory

from qg.extractor import extract_text
from qg.pipeline import generate_questions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB uploads


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/sample")
def sample():
    return send_from_directory(BASE_DIR, "sample_material.txt",
                               mimetype="text/plain")


@app.route("/generate", methods=["POST"])
def generate():
    text = (request.form.get("text") or "").strip()

    up = request.files.get("file")
    if up and up.filename:
        try:
            text = extract_text(up)
        except Exception as exc:
            return jsonify({"error": f"Could not read file: {exc}"}), 400

    types = request.form.getlist("types") or ["mcq", "fib", "tf", "wh"]
    try:
        count = max(1, min(30, int(request.form.get("count", 10))))
    except (TypeError, ValueError):
        count = 10

    if len(text) < 200:
        return jsonify({"error": "Please provide more study material "
                                 "(at least a few sentences)."}), 400
    try:
        questions = generate_questions(text, types=types, count=count)
    except RuntimeError as exc:  # e.g. spaCy model missing
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        return jsonify({"error": f"Generation failed: {exc}"}), 500

    if not questions:
        return jsonify({"error": "Could not generate questions from this "
                                 "text. Try longer material with clear "
                                 "facts."}), 422
    return jsonify({"questions": questions, "count": len(questions)})


if __name__ == "__main__":
    print("QuestGen AI running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
