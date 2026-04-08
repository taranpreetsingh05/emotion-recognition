"""
Dataset Utilities
CSS 2203 – IAI Project | IT-C Group 11

Handles:
  • Synthetic dataset generation (for demo/testing without downloading MELD)
  • Dataset loading / preprocessing helpers
  • Train / test split and evaluation reporting
"""

import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
from model import EMOTIONS, baseline_model, context_model

random.seed(42)
np.random.seed(42)

# ── Synthetic sentence templates per emotion ──────────────────────────────────
TEMPLATES = {
    "joy": [
        "I am so happy today!",
        "This is absolutely wonderful news!",
        "I love spending time with you.",
        "Today was the best day ever!",
        "I feel fantastic, thank you so much!",
        "We finally did it, this is amazing!",
        "I can't stop smiling, everything is perfect.",
        "You made my day, I'm thrilled!",
        "Life is great and I feel blessed.",
        "I'm so excited about our trip!",
    ],
    "sadness": [
        "I feel really down today.",
        "I miss you so much, it hurts.",
        "Everything seems hopeless right now.",
        "I can't stop crying, I'm devastated.",
        "I lost someone very close to me.",
        "Nothing makes me happy anymore.",
        "I feel completely alone in this world.",
        "My heart is broken and I don't know what to do.",
        "I'm so sad I can barely get up.",
        "Life feels empty without you here.",
    ],
    "anger": [
        "I am absolutely furious right now!",
        "How dare you say that to me!",
        "This is completely unacceptable behaviour!",
        "I'm so mad I can barely think straight.",
        "Stop lying to me, I hate this!",
        "You ruined everything with your carelessness!",
        "I've had enough of this nonsense!",
        "This makes my blood boil!",
        "Why can't you just listen for once?",
        "I'm outraged by what just happened.",
    ],
    "fear": [
        "I'm terrified of what might happen next.",
        "Something feels very wrong, I'm scared.",
        "I don't know if I can handle this.",
        "I'm so anxious about tomorrow's results.",
        "The thought of that makes me shudder.",
        "I can't sleep, I keep having nightmares.",
        "I'm worried this will never get better.",
        "I feel a deep sense of dread.",
        "My heart is racing with fear.",
        "Please don't leave me alone right now.",
    ],
    "disgust": [
        "That is absolutely revolting, I can't believe it.",
        "I find that behaviour completely disgusting.",
        "The smell was horrible, I felt sick.",
        "I'm repulsed by what they did.",
        "That is the nastiest thing I've seen.",
        "How can anyone do something so vile?",
        "I'm grossed out and want to leave.",
        "This situation is utterly reprehensible.",
        "I can't stand the sight of that.",
        "Everything about this disgusts me deeply.",
    ],
    "surprise": [
        "Wow, I did not see that coming at all!",
        "That was completely unexpected, I'm shocked!",
        "I can't believe this is actually happening!",
        "No way! Are you serious right now?",
        "That blew my mind, I'm astonished.",
        "I never thought this would happen!",
        "What a twist, nobody expected that!",
        "Oh my goodness, this is unbelievable!",
        "I'm stunned, I have no words.",
        "Wait, really? That's incredible news!",
    ],
    "neutral": [
        "I'll be there around three o'clock.",
        "Could you pass me the report please?",
        "I think we should discuss this later.",
        "The meeting is scheduled for Monday.",
        "Let me know when you are ready.",
        "I need to check on a few things.",
        "That seems like a reasonable approach.",
        "We can look at the options available.",
        "I'll send you the details by email.",
        "The weather today is fairly typical.",
    ],
}

# Augment with minor variations
def _augment(sentences, n=5):
    out = list(sentences)
    for _ in range(n):
        s = random.choice(sentences)
        words = s.split()
        if len(words) > 3:
            i = random.randint(0, len(words) - 1)
            words[i] = words[i]  # identity (can add synonyms in a full impl)
        out.append(" ".join(words))
    return out


def generate_synthetic_dataset(samples_per_emotion: int = 60):
    """
    Returns (texts, labels) lists for training.
    Uses template sentences + light augmentation.
    """
    texts, labels = [], []
    for emo, sentences in TEMPLATES.items():
        augmented = _augment(sentences, n=max(0, samples_per_emotion - len(sentences)))
        for sent in augmented[:samples_per_emotion]:
            texts.append(sent)
            labels.append(emo)
    # Shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


def train_and_evaluate(samples_per_emotion: int = 60):
    """
    Train both models and return evaluation metrics dict.
    """
    texts, labels = generate_synthetic_dataset(samples_per_emotion)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline_model.fit(X_train, y_train)
    b_preds = [baseline_model.predict(t)["emotion"] for t in X_test]
    b_acc  = accuracy_score(y_test, b_preds)
    b_prec = precision_score(y_test, b_preds, average="weighted", zero_division=0)
    b_rec  = recall_score(y_test, b_preds, average="weighted", zero_division=0)
    b_f1   = f1_score(y_test, b_preds, average="weighted", zero_division=0)

    # ── Context-Aware ─────────────────────────────────────────────────────────
    context_model.fit(X_train, y_train)
    c_preds = []
    context_model.reset_context()
    for t in X_test:
        c_preds.append(context_model.predict(t)["emotion"])

    c_acc  = accuracy_score(y_test, c_preds)
    c_prec = precision_score(y_test, c_preds, average="weighted", zero_division=0)
    c_rec  = recall_score(y_test, c_preds, average="weighted", zero_division=0)
    c_f1   = f1_score(y_test, c_preds, average="weighted", zero_division=0)

    b_report = classification_report(y_test, b_preds, output_dict=True, zero_division=0)
    c_report = classification_report(y_test, c_preds, output_dict=True, zero_division=0)

    return {
        "baseline": {
            "accuracy": round(b_acc, 4),
            "precision": round(b_prec, 4),
            "recall": round(b_rec, 4),
            "f1": round(b_f1, 4),
            "report": b_report,
        },
        "context_aware": {
            "accuracy": round(c_acc, 4),
            "precision": round(c_prec, 4),
            "recall": round(c_rec, 4),
            "f1": round(c_f1, 4),
            "report": c_report,
        },
        "samples": {
            "train": len(X_train),
            "test":  len(X_test),
            "emotions": EMOTIONS,
        }
    }


if __name__ == "__main__":
    print("Training models on synthetic dataset …")
    results = train_and_evaluate()
    print("\n── Baseline ──")
    print(f"  Accuracy : {results['baseline']['accuracy']:.4f}")
    print(f"  Precision: {results['baseline']['precision']:.4f}")
    print(f"  Recall   : {results['baseline']['recall']:.4f}")
    print(f"  F1-Score : {results['baseline']['f1']:.4f}")
    print("\n── Context-Aware ──")
    print(f"  Accuracy : {results['context_aware']['accuracy']:.4f}")
    print(f"  Precision: {results['context_aware']['precision']:.4f}")
    print(f"  Recall   : {results['context_aware']['recall']:.4f}")
    print(f"  F1-Score : {results['context_aware']['f1']:.4f}")
