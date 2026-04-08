"""
Flask REST API
CSS 2203 – IAI Project | IT-C Group 11

Endpoints:
  POST /api/predict          – predict emotion for one utterance (both models)
  POST /api/conversation     – multi-turn context-aware prediction
  POST /api/reset            – reset conversation context
  GET  /api/train            – train on synthetic data and return metrics
  GET  /api/health           – health check
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify
from flask_cors import CORS
from model import baseline_model, context_model, EMOTIONS, EMOTION_META
from dataset import train_and_evaluate

app = Flask(__name__)
CORS(app)

# ── Auto-train at startup ─────────────────────────────────────────────────────
_metrics = None

def _ensure_trained():
    global _metrics
    if not baseline_model._trained:
        _metrics = train_and_evaluate()

_ensure_trained()


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "trained": baseline_model._trained})


@app.route("/api/train", methods=["GET"])
def train():
    global _metrics
    _metrics = train_and_evaluate()
    return jsonify({"success": True, "metrics": _metrics})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    b_result = baseline_model.predict(text)
    c_result = context_model.predict(text)

    # Attach metadata
    for result in (b_result, c_result):
        emo = result["emotion"]
        result["emoji"] = EMOTION_META[emo]["emoji"]
        result["color"] = EMOTION_META[emo]["color"]
        result["valence"] = EMOTION_META[emo]["valence"]

    return jsonify({
        "text":          text,
        "baseline":      b_result,
        "context_aware": c_result,
        "emotions":      EMOTIONS,
        "emotion_meta":  EMOTION_META,
    })


@app.route("/api/conversation", methods=["POST"])
def conversation():
    """
    Accepts a list of utterances and returns predictions for each,
    showing how context shifts the emotion predictions over time.
    """
    data = request.get_json(force=True)
    utterances = data.get("utterances", [])
    if not utterances:
        return jsonify({"error": "utterances list is required"}), 400

    context_model.reset_context()
    results = []
    for utt in utterances:
        text = utt.strip()
        if not text:
            continue
        b = baseline_model.predict(text)
        c = context_model.predict(text)
        results.append({
            "text":          text,
            "baseline":      {**b, "emoji": EMOTION_META[b["emotion"]]["emoji"],
                               "color": EMOTION_META[b["emotion"]]["color"]},
            "context_aware": {**c, "emoji": EMOTION_META[c["emotion"]]["emoji"],
                               "color": EMOTION_META[c["emotion"]]["color"]},
        })

    return jsonify({"turns": results, "emotion_meta": EMOTION_META})


@app.route("/api/reset", methods=["POST"])
def reset_context():
    context_model.reset_context()
    return jsonify({"success": True, "message": "Conversation context cleared."})


@app.route("/api/metrics", methods=["GET"])
def metrics():
    if _metrics is None:
        return jsonify({"error": "not trained yet"}), 400
    return jsonify(_metrics)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
