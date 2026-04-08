"""
Context-Aware Emotion Recognition Model
CSS 2203 – IAI Project | IT-C Group 11

This module implements both:
1. Baseline Model  – classifies each utterance independently
2. Context-Aware Model – uses conversational history for temporal consistency
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from collections import deque
import re

# ── Emotion labels (aligned with MELD / DailyDialog conventions) ──────────────
EMOTIONS = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]

EMOTION_META = {
    "neutral":  {"emoji": "😐", "color": "#6B7280", "valence": 0.0},
    "joy":      {"emoji": "😊", "color": "#F59E0B", "valence": 1.0},
    "sadness":  {"emoji": "😢", "color": "#3B82F6", "valence": -0.8},
    "anger":    {"emoji": "😠", "color": "#EF4444", "valence": -0.9},
    "fear":     {"emoji": "😨", "color": "#8B5CF6", "valence": -0.7},
    "disgust":  {"emoji": "🤢", "color": "#10B981", "valence": -0.6},
    "surprise": {"emoji": "😲", "color": "#EC4899", "valence": 0.3},
}

# ── Keyword-augmented lexicon for richer feature signals ──────────────────────
EMOTION_LEXICON = {
    "joy":      ["happy", "great", "love", "wonderful", "excited", "fantastic",
                 "awesome", "delighted", "thrilled", "joyful", "glad", "pleased"],
    "sadness":  ["sad", "unhappy", "depressed", "miss", "lonely", "cry", "grief",
                 "heartbroken", "sorrow", "gloomy", "hopeless", "devastated"],
    "anger":    ["angry", "furious", "hate", "mad", "rage", "frustrated",
                 "annoyed", "livid", "outraged", "infuriated", "irritated"],
    "fear":     ["scared", "afraid", "terrified", "worried", "anxious", "dread",
                 "nervous", "horrified", "panic", "frightened", "uneasy"],
    "disgust":  ["disgusting", "gross", "nasty", "revolting", "awful",
                 "horrible", "repulsive", "vile", "yuck", "filthy"],
    "surprise": ["wow", "amazing", "unbelievable", "shocking", "incredible",
                 "unexpected", "astonished", "stunned", "whoa", "really"],
    "neutral":  ["okay", "fine", "alright", "sure", "maybe", "perhaps",
                 "think", "know", "want", "need"],
}


# ─────────────────────────────────────────────────────────────────────────────
#  Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def _lexicon_features(text: str) -> np.ndarray:
    """Return a 7-dim vector of normalised keyword hit-counts per emotion."""
    text_lower = text.lower()
    feats = []
    for emo in EMOTIONS:
        hits = sum(1 for kw in EMOTION_LEXICON.get(emo, []) if kw in text_lower)
        feats.append(hits)
    total = max(sum(feats), 1)
    return np.array(feats, dtype=float) / total


def _punctuation_features(text: str) -> np.ndarray:
    """Capture exclamation/question marks and CAPS ratio as auxiliary signals."""
    excl = text.count("!")
    quest = text.count("?")
    caps = sum(1 for c in text if c.isupper())
    ratio = caps / max(len(text), 1)
    return np.array([min(excl, 5), min(quest, 5), ratio], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline Model (no context)
# ─────────────────────────────────────────────────────────────────────────────

class BaselineEmotionModel:
    """
    Predicts emotion from a single utterance using TF-IDF + Logistic Regression.
    Serves as the performance benchmark.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000,
                                      sublinear_tf=True)),
            ("clf",   LogisticRegression(max_iter=1000, C=1.0,
                                          class_weight="balanced",
                                          random_state=42)),
        ])
        self._trained = False

    def fit(self, texts, labels):
        self.pipeline.fit(texts, labels)
        self._trained = True

    def predict(self, text: str) -> dict:
        if not self._trained:
            return self._rule_based_fallback(text)
        probs = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        emotion = classes[np.argmax(probs)]
        confidence = float(np.max(probs))
        all_probs = {cls: float(p) for cls, p in zip(classes, probs)}
        return {"emotion": emotion, "confidence": confidence, "probs": all_probs}

    def _rule_based_fallback(self, text: str) -> dict:
        """Used when model has not been trained yet (demo mode)."""
        lex = _lexicon_features(text)
        punct = _punctuation_features(text)
        # Boost joy/anger with punctuation
        lex[EMOTIONS.index("joy")]   += punct[0] * 0.3
        lex[EMOTIONS.index("anger")] += punct[0] * 0.2
        idx = int(np.argmax(lex)) if lex.max() > 0 else 0
        emotion = EMOTIONS[idx]
        probs = {e: float(v) for e, v in zip(EMOTIONS, lex / max(lex.sum(), 1e-6))}
        return {"emotion": emotion, "confidence": float(lex[idx]), "probs": probs}


# ─────────────────────────────────────────────────────────────────────────────
#  Context-Aware Model (with temporal consistency)
# ─────────────────────────────────────────────────────────────────────────────

class ContextAwareEmotionModel:
    """
    Extends the baseline with a sliding-window conversational context buffer.

    Temporal Consistency Mechanism:
    --------------------------------
    After predicting an emotion for the current utterance, the model applies a
    smoothing step that penalises abrupt transitions using a learned or heuristic
    transition matrix.  This mirrors how human emotions evolve gradually in
    natural conversation rather than jumping erratically between poles.
    """

    # Heuristic transition cost matrix  (from_emotion → to_emotion → penalty)
    TRANSITION_COST = {
        "joy":     {"anger": 0.4, "disgust": 0.3, "fear": 0.2, "sadness": 0.3},
        "sadness": {"joy": 0.3, "anger": 0.2, "surprise": 0.2},
        "anger":   {"joy": 0.4, "surprise": 0.1},
        "fear":    {"joy": 0.3, "anger": 0.1},
        "neutral": {},   # neutral transitions cheaply to anything
        "disgust": {"joy": 0.3, "surprise": 0.2},
        "surprise": {"disgust": 0.2},
    }

    def __init__(self, context_window: int = 3, smoothing_alpha: float = 0.25):
        self.baseline = BaselineEmotionModel()
        self.context_window = context_window
        self.smoothing_alpha = smoothing_alpha           # weight for context penalty
        self.history: deque = deque(maxlen=context_window)
        self._trained = False

    def fit(self, texts, labels):
        self.baseline.fit(texts, labels)
        self._trained = True

    def reset_context(self):
        """Clear the conversation history (call at the start of a new conversation)."""
        self.history.clear()

    def _build_context_text(self, current_text: str) -> str:
        """Concatenate recent history with the current utterance."""
        parts = list(self.history) + [current_text]
        return " [SEP] ".join(parts[-self.context_window:])

    def _apply_temporal_smoothing(self, probs: dict) -> dict:
        """
        Penalise emotionally abrupt transitions based on previous turn's emotion.
        """
        if not self.history:
            return probs

        # Infer previous emotion from history (last predicted)
        prev = getattr(self, "_last_emotion", "neutral")
        penalties = self.TRANSITION_COST.get(prev, {})

        adjusted = {}
        for emo, p in probs.items():
            penalty = penalties.get(emo, 0.0)
            adjusted[emo] = p * (1 - self.smoothing_alpha * penalty)

        # Renormalise
        total = sum(adjusted.values()) or 1e-6
        return {e: v / total for e, v in adjusted.items()}

    def predict(self, text: str) -> dict:
        # 1. Get base probabilities from current utterance alone (accurate signal)
        if self._trained:
            raw = self.baseline.pipeline.predict_proba([text])[0]
            classes = self.baseline.pipeline.classes_
            probs = {cls: float(p) for cls, p in zip(classes, raw)}
        else:
            result = self.baseline._rule_based_fallback(text)
            probs = result["probs"]

        # 2. Apply temporal smoothing (adjusts based on conversation history)
        smoothed = self._apply_temporal_smoothing(probs)

        # 4. Final prediction
        emotion = max(smoothed, key=smoothed.get)
        confidence = smoothed[emotion]

        # 5. Update state
        self._last_emotion = emotion
        self.history.append(text)

        return {
            "emotion":    emotion,
            "confidence": round(confidence, 4),
            "probs":      {e: round(v, 4) for e, v in smoothed.items()},
            "context_used": list(self.history)[:-1],   # history BEFORE this turn
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Shared singleton instances  (imported by app.py)
# ─────────────────────────────────────────────────────────────────────────────

baseline_model = BaselineEmotionModel()
context_model  = ContextAwareEmotionModel(context_window=3, smoothing_alpha=0.25)
