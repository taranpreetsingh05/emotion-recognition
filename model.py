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
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report, accuracy_score
from collections import deque
import re
def detect_aggressive_tone(text):
    text = text.lower()
    patterns = [
        "are you crazy",
        "what is wrong with you",
        "this is not acceptable",
        "this is ridiculous",
        "this is insane"
    ]
    return any(p in text for p in patterns)
# ── Emotion labels ─────────────────────────────────────────────────────────────
EMOTIONS = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise", "love"]

EMOTION_META = {
    "neutral":  {"emoji": "😐", "color": "#6B7280", "valence":  0.0},
    "joy":      {"emoji": "😊", "color": "#F59E0B", "valence":  1.0},
    "sadness":  {"emoji": "😢", "color": "#3B82F6", "valence": -0.8},
    "anger":    {"emoji": "😠", "color": "#EF4444", "valence": -0.9},
    "fear":     {"emoji": "😨", "color": "#8B5CF6", "valence": -0.7},
    "disgust":  {"emoji": "🤢", "color": "#10B981", "valence": -0.6},
    "surprise": {"emoji": "😲", "color": "#EC4899", "valence":  0.3},
    "love":     {"emoji": "❤️",  "color": "#F43F5E", "valence":  0.95},
}

# ── Words that are ONLY meaningful in emotional context, not standalone ─────────
# These appear in many emotion sentences but carry zero standalone signal.
# Removed from lexicon to prevent pronoun/question-word contamination.
_AMBIGUOUS_STOP = {
    "you", "how", "what", "why", "when", "i", "me", "my", "we", "our",
    "this", "that", "it", "they", "them", "he", "she", "is", "was",
    "are", "were", "do", "did", "have", "has", "had", "going", "gonna",
    "will", "would", "could", "should", "can", "may", "might", "must",
}

# ── Greeting / social phrases → always neutral ────────────────────────────────
GREETING_PATTERNS = [
    r"^(hi|hey|hello|howdy|hiya|yo|sup|heya|helo|hii+|heyy+|hiiii*)[\s!?.]*$",
    r"^(good\s+(morning|afternoon|evening|night|day))[\s!?.]*$",
    r"^(how are you|how r u|how are u|how do you do|how's it going|how's everything|"
     r"how are things|what's up|wassup|wazzup|whats up|what is up)[\s!?.]*$",
    r"^(nice to meet you|pleased to meet you|good to see you)[\s!?.]*$",
    r"^(bye|goodbye|see you|see ya|take care|later|cya|ttyl|talk later)[\s!?.]*$",
    r"^(okay|ok|sure|alright|fine|yep|nope|yes|no|maybe|perhaps)[\s!?.]*$",
    r"^(thanks|thank you|ty|thx|cheers|welcome|you're welcome)[\s!?.]*$",
]

def _is_greeting(text: str) -> bool:
    """Returns True if the text is a pure greeting / social filler."""
    t = text.strip().lower()
    for pat in GREETING_PATTERNS:
        if re.match(pat, t):
            return True
    return False


# ── Emotion lexicon ────────────────────────────────────────────────────────────
# RULES:
#  1. No bare pronouns or question words (how, you, what, why) — they appear in
#     ALL emotion classes and pollute TF-IDF coefficients.
#  2. "love" uses multi-word anchors only, never bare "love" (too ambiguous).
#  3. Anger includes threat / violent-action vocabulary.
#  4. Neutral has strong coverage for greetings and factual language.
EMOTION_LEXICON = {
    "neutral": [
        # Greetings & social
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
        "how are you", "how's it going", "what's up", "nice to meet you",
        "bye", "goodbye", "see you", "take care", "thanks", "thank you",
        # Factual / task language
        "okay", "alright", "sure", "maybe", "perhaps", "fine",
        "meeting", "schedule", "later", "tomorrow", "today", "report",
        "email", "call", "check", "send", "seems", "reasonable",
        "let me know", "could you", "would you", "please confirm",
        "i think", "we should", "the plan", "the meeting", "as discussed",
    ],
    "joy": [
        "happy", "happiness", "wonderful", "excited", "fantastic",
        "awesome", "delighted", "thrilled", "joyful", "glad", "pleased",
        "cheerful", "elated", "jubilant", "ecstatic", "overjoyed",
        "blissful", "grateful", "lucky", "blessed", "brilliant",
        "excellent", "perfect", "terrific", "celebrate", "celebration",
        "smiling", "laughing", "fun", "enjoy", "enjoying",
        "hooray", "yay", "woohoo", "so happy", "best day",
    ],
    "sadness": [
        "sad", "sadness", "unhappy", "depressed", "depression",
        "missing someone", "lonely", "loneliness", "crying", "grief",
        "grieve", "heartbroken", "heartbreak", "sorrow", "sorrowful",
        "gloomy", "hopeless", "devastated", "despair", "miserable",
        "misery", "alone", "abandoned", "painful", "aching", "tears",
        "weeping", "mourning", "regret", "disappointed", "disappointment",
        "tragic", "tragedy", "falling apart", "giving up", "feel empty",
        "feel lost", "feel hopeless","exhausted"
    ],
    "anger": [
        # Emotional anger words
        "angry", "furious", "fury", "hatred", "rage", "raging",
        "frustrated", "frustration", "annoyed", "livid", "outraged",
        "infuriated", "irritated", "hostile", "aggressive", "violent",
        "screaming", "cursing", "damn it", "stupid", "idiot", "moron",
        "fed up", "unacceptable", "ridiculous", "rubbish", "nonsense",
        "cheating", "betrayed", "betrayal", "injustice", "unfair",
        "i hate", "i'm furious", "makes me mad", "so angry",
        # Threat & violent-action vocabulary
        "rip apart", "rip you", "tear apart", "gonna kill",
        "gonna destroy", "gonna beat", "gonna hurt", "gonna rip",
        "gonna smash", "gonna tear", "going to kill", "going to hurt",
        "going to destroy", "going to beat", "will kill", "will hurt",
        "will destroy", "make you pay", "make you suffer", "you're dead",
        "you are dead", "watch yourself", "last warning", "warning you",
        "i swear", "come for you", "coming for you", "finish you",
        "end you", "smash you", "beat you", "crush you", "murder you",
        "strangle you", "choke you", "punch you", "hit you", "hurt you",
        "unacceptable", "ridiculous", "crazy", "insane",
        "not acceptable", "not okay", "nonsense"
    ],
    "fear": [
        "scared", "terrified", "terror", "worried", "anxious",
        "anxiety", "dreading", "nervous", "nervousness", "horrified",
        "panicking", "frightened", "uneasy", "apprehensive",
        "threatened", "danger", "dangerous", "nightmares",
        "shaking", "trembling", "paranoid", "insecure", "helpless",
        "ominous", "foreboding", "creepy", "terrifying", "alarmed",
        "heart racing", "can't breathe", "so scared", "so afraid",
        "i'm terrified", "i'm afraid",
    ],
    "disgust": [
        "disgusting", "disgusted", "gross", "nasty", "revolting",
        "repulsive", "vile", "yuck", "filthy", "filth", "dirty",
        "sickening", "nauseating", "nausea", "repelled",
        "offensive", "appalling", "appalled", "despicable",
        "loathe", "loathing", "abhorrent", "contempt",
        "putrid", "rotten", "stench", "foul", "wretched",
        "atrocious", "obscene", "vulgar", "crude", "distasteful",
        "i'm repulsed", "makes me sick", "can't stand the sight",
    ],
    "surprise": [
        "unbelievable", "shocking", "shocked", "incredible",
        "unexpected", "astonished", "stunned", "whoa",
        "no way", "oh my gosh", "oh my god", "omg",
        "goodness", "gosh", "sudden", "suddenly",
        "startled", "mindblowing", "extraordinary", "remarkable",
        "speechless", "breathtaking", "astounding", "jaw dropped",
        "i can't believe", "never expected", "never thought",
        "blew my mind",
    ],
    "love": [
        # Multi-word anchors only — bare "love" is too ambiguous
        "love you", "i love", "in love with", "fell in love",
        "adore you", "adore him", "adore her", "adoration",
        "cherish you", "cherish him", "cherish her",
        "affectionate", "devoted to", "devotion to",
        "romantic feelings", "romance with",
        "darling", "sweetheart", "honey", "beloved",
        "soulmate", "infatuated", "crush on",
        "miss you so", "need you so", "head over heels",
        "you mean everything", "you mean the world",
        "my heart belongs", "forever with you",
        "kissing you", "hugging you", "cuddling with",
    ],
}

# ── High-confidence phrase patterns (score ×4 each match) ─────────────────────
PHRASE_PATTERNS = {
    "neutral": [
        r"^(hi|hey|hello|hiya|yo)[\s!?.]*$",
        r"^(good\s+(morning|afternoon|evening|night|day))[\s!?.]*$",
        r"how are (you|u|things|everything)",
        r"(nice|good|great) to (meet|see|hear from) you",
        r"(what's|whats) up",
        r"(let me|i will|i'll) (check|confirm|send|get back)",
        r"(the meeting|the call|the report|the schedule)",
        r"^(okay|ok|sure|alright|thanks|thank you|bye|goodbye)[\s!?.]*$",
    ],
    "anger": [
        r"i('m| am) (so |absolutely |really )?(furious|livid|outraged|enraged)",
        r"how dare you",
        r"makes my blood boil",
        r"(fed up|sick and tired) with",
        r"drives me (crazy|nuts|mad|insane)",
        r"(gonna|going to|am going to|will) (rip|kill|destroy|hurt|beat|smash|crush|murder|end|finish) (you|him|her|them|it)",
        r"(rip|tear) (you |him |her |them )?(apart|to pieces)",
        r"i (swear|promise) (i will|i'll|to god)",
        r"you('re| are) (dead|finished|done|over)",
        r"make you (pay|suffer|regret)",
        r"(come for you|coming for you)",
        r"i (hate|can't stand) (you|this|him|her|them)",
        r"i'?m (so |really )?angry",
        r"not acceptable",
        r"are you crazy",
        r"what is wrong with you",
        r"this is unacceptable",
        r"this is ridiculous",
        r"this is not okay",
        r"this is insane",
    ],
    "love": [
        r"i love you",
        r"i('m| am) in love (with you)?",
        r"you mean (everything|the world) to me",
        r"head over heels (for you)?",
        r"can't stop thinking about you",
        r"you('re| are) my (everything|world|life|heart|soulmate)",
        r"fall(ing)? in love (with you)?",
        r"(deeply|madly|truly) in love",
        r"i adore you",
        r"i cherish you",
    ],
    "joy": [
        r"over the moon",
        r"on top of the world",
        r"best day (ever|of my life)",
        r"(so|really|absolutely) happy",
        r"couldn't be (happier|better)",
        r"(absolutely|truly) (wonderful|amazing|fantastic|brilliant)",
        r"i('m| am) (so |really )?(happy|thrilled|excited|elated|overjoyed)",
    ],
    "sadness": [
        r"i('m| am) (so |really |absolutely )?(sad|devastated|heartbroken|depressed)",
        r"can't stop (crying|sobbing)",
        r"(falling|fallen) apart",
        r"feel(ing)? (so )?(empty|hopeless|alone|lost)",
        r"nothing (matters|makes me happy) anymore",
        r"i give up",
        r"(i don't|i no longer) (love|care about|feel) (you|him|her|them|anything)",
    ],
    "fear": [
        r"i('m| am) (so |really |absolutely )?(scared|terrified|afraid|petrified)",
        r"can't sleep (because|from|due to)",
        r"(sense of|filled with) dread",
        r"heart (is )?racing",
        r"(having )?panic attack",
        r"i('m| am) (so |really )?worried about",
    ],
    "surprise": [
        r"(oh my god|oh my gosh|omg|oh wow)",
        r"i can't believe (this|it|that|what)",
        r"no (way|freaking way)",
        r"are you (serious|kidding|joking)",
        r"(blew|blown) my mind",
        r"never (expected|thought|imagined) (this|that)",
    ],
    "disgust": [
        r"(absolutely|totally|completely) (disgusting|revolting|repulsive|vile)",
        r"makes me (sick|nauseous|want to vomit)",
        r"can't stand the (sight|smell|thought) of",
        r"(so |utterly )?repulsed by",
        r"i('m| am) (disgusted|appalled|revolted)",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
#  Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def _lexicon_features(text: str) -> np.ndarray:
    """Normalised keyword hit-count vector (one dim per emotion)."""
    t = text.lower()
    feats = [sum(1 for kw in EMOTION_LEXICON.get(emo, []) if kw in t)
             for emo in EMOTIONS]
    total = max(sum(feats), 1)
    return np.array(feats, dtype=float) / total


def _phrase_features(text: str) -> np.ndarray:
    """Phrase-pattern hit-count vector — 4× weight per match."""
    t = text.lower()
    feats = [sum(6 for pat in PHRASE_PATTERNS.get(emo, []) if re.search(pat, t))
             for emo in EMOTIONS]
    total = max(sum(feats), 1)
    return np.array(feats, dtype=float) / total


def _punctuation_features(text: str) -> np.ndarray:
    """Exclamation marks, question marks, ALL-CAPS ratio, heart symbols."""
    excl      = min(text.count("!"), 5)
    quest     = min(text.count("?"), 5)
    caps      = sum(1 for c in text if c.isupper())
    cap_ratio = caps / max(len(text), 1)
    heart     = 1 if any(s in text for s in ["<3", "❤", "♥", "💕", "💗", "💓", "💖"]) else 0
    return np.array([excl, quest, cap_ratio, heart], dtype=float)


def _negation_features(text: str) -> np.ndarray:
    """Penalise love/joy when negated (e.g. 'don't love you', 'not happy')."""
    t          = text.lower()
    penalties  = np.zeros(len(EMOTIONS), dtype=float)
    neg        = r"(don'?t|do not|doesn'?t|does not|didn'?t|did not|never|no longer|not)\s+"
    love_idx   = EMOTIONS.index("love")
    joy_idx    = EMOTIONS.index("joy")

    if re.search(neg + r"(love|adore|care about|like|need|want) (you|him|her|them)", t):
        penalties[love_idx] = 0.7

    if re.search(neg + r"(happy|like|enjoy|want)", t):
        penalties[joy_idx] = 0.5

    return penalties


def _greeting_boost(text: str) -> np.ndarray:
    """Give a strong neutral boost for greetings and simple social phrases."""
    boost = np.zeros(len(EMOTIONS), dtype=float)
    if _is_greeting(text):
        boost[EMOTIONS.index("neutral")] = 5.0
    return boost


class LexiconTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer combining all hand-crafted features."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        rows = []
        for text in X:
            lex      = _lexicon_features(text)
            phrase   = _phrase_features(text)
            punct    = _punctuation_features(text)
            neg      = _negation_features(text)
            greeting = _greeting_boost(text)
            rows.append(np.concatenate([lex, phrase, punct, neg, greeting]))
        arr = np.array(rows, dtype=float)
        return normalize(arr, norm="l2")


# ─────────────────────────────────────────────────────────────────────────────
#  Custom TF-IDF that ignores ambiguous pronouns/question-words
# ─────────────────────────────────────────────────────────────────────────────

# Stop words that pollute emotion classification.
# Standard English stopwords PLUS pronouns and question words that appear
# in all emotion classes and destroy the classifier's discrimination.
_TFIDF_STOPWORDS = list({
    # Standard English function words
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "can", "may", "might", "must", "shall",
    # Pronouns — appear in ALL emotion classes equally
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    # Question words — appear in anger ("how dare you") AND neutral ("how are you")
    "how", "what", "why", "when", "where", "who", "which",
    # Other high-frequency ambiguous words
    "this", "that", "these", "those", "there", "here", "so", "just",
    "now", "then", "also", "very", "too", "up", "out", "about",
    "into", "than", "more", "some", "such", "like", "even", "no",
    "not", "only", "same", "other", "s", "t", "re", "ve", "ll",
})


# ─────────────────────────────────────────────────────────────────────────────
#  Baseline Model (no context)
# ─────────────────────────────────────────────────────────────────────────────

class BaselineEmotionModel:
    """TF-IDF (filtered stopwords) + hand-crafted features → Logistic Regression."""

    def __init__(self):
        combined = FeatureUnion([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=10000,
                sublinear_tf=True,
                min_df=1,
                analyzer="word",
                token_pattern=r"(?u)\b\w+\b",
                stop_words=list(_TFIDF_STOPWORDS),
            )),
            ("lexicon", LexiconTransformer()),
        ])
        self.pipeline = Pipeline([
            ("features", combined),
            ("clf", LogisticRegression(
                max_iter=3000,
                C=3.0,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            )),
        ])
        self._trained = False

    def fit(self, texts, labels):
        self.pipeline.fit(texts, labels)
        self._trained = True

    def predict(self, text: str) -> dict:

    # ✅ 1. Aggressive tone check (FIRST)
     if detect_aggressive_tone(text):
        return {
            "emotion": "anger",
            "confidence": 0.85,
            "probs": {e: 0.0 for e in EMOTIONS}
        }

    # ✅ 2. Greeting override
     if _is_greeting(text):
        probs = {e: 0.02 for e in EMOTIONS}
        probs["neutral"] = 0.84
        return {"emotion": "neutral", "confidence": 0.84, "probs": probs}

    # ✅ 3. If not trained
     if not self._trained:
        return self._rule_based_fallback(text)

    # ✅ 4. Normal prediction
     probs_arr = self.pipeline.predict_proba([text])[0]
     classes = self.pipeline.classes_

     emotion = classes[np.argmax(probs_arr)]
     confidence = float(np.max(probs_arr))
     all_probs = {cls: float(p) for cls, p in zip(classes, probs_arr)}

     return {"emotion": emotion, "confidence": confidence, "probs": all_probs}
    def _rule_based_fallback(self, text: str) -> dict:
        """Rule-based prediction used before training completes."""
        if _is_greeting(text):
            probs = {e: 0.02 for e in EMOTIONS}
            probs["neutral"] = 0.84
            return {"emotion": "neutral", "confidence": 0.84, "probs": probs}

        lex    = _lexicon_features(text)
        phrase = _phrase_features(text)
        neg    = _negation_features(text)
        greet  = _greeting_boost(text)
        punct  = _punctuation_features(text)

        scores = lex + phrase + greet
        scores = np.maximum(scores - neg, 0)
        scores[EMOTIONS.index("anger")] += punct[0] * 0.12
        scores[EMOTIONS.index("joy")]   += punct[0] * 0.08
        scores[EMOTIONS.index("love")]  += punct[3] * 0.6

        total = scores.sum()
        if total > 0:
            scores /= total
        else:
            scores[EMOTIONS.index("neutral")] = 1.0

        idx     = int(np.argmax(scores))
        emotion = EMOTIONS[idx]
        probs   = {e: float(v) for e, v in zip(EMOTIONS, scores)}
        return {"emotion": emotion, "confidence": float(scores[idx]), "probs": probs}


# ─────────────────────────────────────────────────────────────────────────────
#  Context-Aware Model
# ─────────────────────────────────────────────────────────────────────────────

class ContextAwareEmotionModel:
    """
    Sliding-window context buffer + temporal-consistency smoothing.
    Emotionally abrupt transitions are penalised.
    """

    TRANSITION_COST = {
        "joy":      {"anger": 0.4, "disgust": 0.3, "fear": 0.2, "sadness": 0.3},
        "sadness":  {"joy": 0.3, "anger": 0.2, "surprise": 0.2, "love": 0.1},
        "anger":    {"joy": 0.4, "surprise": 0.1, "love": 0.5},
        "fear":     {"joy": 0.3, "anger": 0.1, "love": 0.2},
        "neutral":  {},
        "disgust":  {"joy": 0.3, "surprise": 0.2, "love": 0.4},
        "surprise": {"disgust": 0.2},
        "love":     {"anger": 0.5, "disgust": 0.5, "fear": 0.2},
    }

    def __init__(self, context_window: int = 3, smoothing_alpha: float = 0.25):
        self.baseline        = BaselineEmotionModel()
        self.context_window  = context_window
        self.smoothing_alpha = smoothing_alpha
        self.history: deque  = deque(maxlen=context_window)
        self._trained        = False
        self._last_emotion   = "neutral"

    def fit(self, texts, labels):
        self.baseline.fit(texts, labels)
        self._trained = True

    def reset_context(self):
        self.history.clear()
        self._last_emotion = "neutral"

    def _apply_temporal_smoothing(self, probs: dict) -> dict:
        penalties = self.TRANSITION_COST.get(self._last_emotion, {})
        adjusted  = {emo: p * (1 - self.smoothing_alpha * penalties.get(emo, 0.0))
                     for emo, p in probs.items()}
        total = sum(adjusted.values()) or 1e-6
        return {e: v / total for e, v in adjusted.items()}

    def predict(self, text: str) -> dict:
        # Hard override for greetings
        if _is_greeting(text):
            probs = {e: 0.02 for e in EMOTIONS}
            probs["neutral"] = 0.84
            self._last_emotion = "neutral"
            self.history.append(text)
            return {
                "emotion": "neutral", "confidence": 0.84,
                "probs": probs, "context_used": list(self.history)[:-1],
            }

        if self._trained:
            raw   = self.baseline.pipeline.predict_proba([text])[0]
            probs = {cls: float(p)
                     for cls, p in zip(self.baseline.pipeline.classes_, raw)}
        else:
            probs = self.baseline._rule_based_fallback(text)["probs"]

        smoothed   = self._apply_temporal_smoothing(probs)
        emotion    = max(smoothed, key=smoothed.get)
        confidence = smoothed[emotion]

        self._last_emotion = emotion
        self.history.append(text)

        return {
            "emotion":      emotion,
            "confidence":   round(confidence, 4),
            "probs":        {e: round(v, 4) for e, v in smoothed.items()},
            "context_used": list(self.history)[:-1],
        }


# ── Singleton instances (imported by app.py) ──────────────────────────────────
baseline_model = BaselineEmotionModel()
context_model  = ContextAwareEmotionModel(context_window=3, smoothing_alpha=0.25)
