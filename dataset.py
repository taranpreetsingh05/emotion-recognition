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

# ── Rich synthetic sentence templates per emotion ──────────────────────────────
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
        "This is the happiest I have ever felt!",
        "I'm overjoyed, I can't believe this is real!",
        "Everything is going perfectly, I'm ecstatic!",
        "We won! I'm jumping with joy right now!",
        "Best news ever, I'm absolutely delighted!",
        "I'm on top of the world today!",
        "Hooray, we finally made it happen!",
        "I'm grinning from ear to ear right now.",
        "This celebration is just perfect, I'm elated!",
        "I feel so grateful and cheerful today.",
    ],
    "sadness": [
        "I feel really down today.",
        "I miss you so much, it hurts.",
        "I don't love you anymore, it is over.",
        "I no longer feel anything for you.",
        "I stopped loving you a long time ago.",
        "Everything seems hopeless right now.",
        "I can't stop crying, I'm devastated.",
        "I lost someone very close to me.",
        "Nothing makes me happy anymore.",
        "I feel completely alone in this world.",
        "My heart is broken and I don't know what to do.",
        "I'm so sad I can barely get up.",
        "Life feels empty without you here.",
        "I'm falling apart and no one seems to care.",
        "I feel hopeless and I don't know how to go on.",
        "Everything I worked for is gone, I'm devastated.",
        "I keep crying and I don't even know why.",
        "I feel so lost and empty inside.",
        "Nothing will ever be the same again.",
        "I wish things could have been different.",
        "I'm deeply saddened by what happened.",
        "The grief is overwhelming and I feel helpless.",
        "I'm mourning and there is no comfort anywhere.",
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
        "This is absolutely ridiculous and I'm furious!",
        "I can't believe how irresponsible that was!",
        "That's it, I'm done putting up with this!",
        "How could you betray me like that?",
        "I'm livid and I want answers right now!",
        "You have no right to treat me this way!",
        "I'm fed up and I'm not taking this anymore!",
        "Your behaviour is infuriating and unacceptable!",
        "I'm screaming inside because of what you did.",
        "Stop wasting my time with your nonsense!",
        # Threat / colloquial anger sentences
        "This is not acceptable at all.",
        "Are you crazy or what?",
        "What is wrong with you?",
        "This is ridiculous behaviour.",
        "This makes no sense and I hate it.",
        "I am gonna rip you apart!",
        "I'm going to destroy you, just watch!",
        "I will tear you apart if you do that again!",
        "You better watch yourself, I am warning you!",
        "I am going to kill you for this!",
        "I swear I'm going to hurt you!",
        "I'm going to beat you to a pulp!",
        "You're dead meat when I get my hands on you!",
        "I'll smash everything if you don't stop!",
        "I'm gonna make you pay for what you did!",
        "You are finished, I will end you!",
        "I hate you so much I could scream!",
        "Don't push me or I'm going to lose it!",
        "I want to punch something right now I am so angry!",
        "I'm going to come for you and you won't see it coming!",
        "I could strangle you right now!",
        "I'm furious and I swear to god I'll make you regret this!",
        "Get out before I do something I'll regret!",
        "You have no idea what I'm capable of when I'm this angry!",
        "I'm going to tear this whole thing apart!",
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
        "I'm absolutely terrified of what comes next.",
        "I feel so unsafe and I don't know why.",
        "The anxiety is overwhelming and I can't breathe.",
        "I'm shaking and I can't seem to calm down.",
        "What if everything goes wrong tomorrow?",
        "I'm scared something terrible is about to happen.",
        "I've been having panic attacks every night.",
        "I'm afraid and I need someone to help me.",
        "This situation fills me with dread.",
        "I'm nervous and I can't shake this feeling.",
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
        "That is absolutely foul and I feel nauseated.",
        "How can people behave in such a disgusting way?",
        "I'm appalled by the filth in this place.",
        "I loathe everything about that person's actions.",
        "That smell is absolutely putrid and repulsive.",
        "I'm sickened by how cruel people can be.",
        "The whole thing is vile and I want to leave.",
        "I find that content deeply offensive and gross.",
        "My stomach turns just thinking about it.",
        "This is the most repulsive thing I have ever seen.",
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
        "Oh my gosh, this is absolutely shocking!",
        "I'm completely speechless, I cannot believe this.",
        "Nobody could have predicted this would happen!",
        "My jaw dropped when I heard the news.",
        "That is the most unexpected thing I've ever seen.",
        "Whoa, I had absolutely no idea about this!",
        "Seriously? I'm totally caught off guard here!",
        "I'm astonished, this is beyond remarkable.",
        "I never in a million years expected this.",
        "This revelation is absolutely mind-blowing to me.",
    ],
    "love": [
        "I love you so much, you mean the world to me.",
        "You are the most wonderful person in my life.",
        "I cherish every single moment I spend with you.",
        "My heart is so full of love for you.",
        "I adore you and everything about you.",
        "Being with you is the best feeling in the world.",
        "I'm completely head over heels for you.",
        "You are my everything and I can't imagine life without you.",
        "I fall more in love with you every single day.",
        "You make my heart feel so warm and happy.",
        "I treasure you more than words can ever express.",
        "My love for you grows stronger every day.",
        "I'm so deeply in love with you, my darling.",
        "You are the love of my life and my soulmate.",
        "Thinking of you fills my heart with so much warmth.",
        "I can't stop thinking about you, I'm smitten.",
        "You mean absolutely everything to me.",
        "I feel so blessed to love you and be loved by you.",
        "Every moment with you is a precious gift to me.",
        "I would do anything for you, I love you deeply.",
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
        "Please confirm the time for the call.",
        "I've reviewed the document you sent over.",
        "We should follow the usual procedure here.",
        "Let me get back to you on that shortly.",
        "The schedule has been updated for this week.",
        "I'll forward the relevant information to you.",
        "We need to finalize the plan before Friday.",
        "The team is available to meet tomorrow morning.",
        "I have noted your comments in the file.",
        "Please let me know if you need anything else.",
    ],
}


def _augment(sentences, n=5):
    """Light augmentation: duplicate with minor word-level no-ops."""
    out = list(sentences)
    for _ in range(n):
        s = random.choice(sentences)
        out.append(s)
    return out


def generate_synthetic_dataset(samples_per_emotion: int = 80):
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
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


def train_and_evaluate(samples_per_emotion: int = 80):
    """Train both models and return evaluation metrics dict."""
    texts, labels = generate_synthetic_dataset(samples_per_emotion)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline_model.fit(X_train, y_train)
    b_preds = [baseline_model.predict(t)["emotion"] for t in X_test]
    b_acc   = accuracy_score(y_test, b_preds)
    b_prec  = precision_score(y_test, b_preds, average="weighted", zero_division=0)
    b_rec   = recall_score(y_test, b_preds, average="weighted", zero_division=0)
    b_f1    = f1_score(y_test, b_preds, average="weighted", zero_division=0)

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
            "accuracy":  round(b_acc, 4),
            "precision": round(b_prec, 4),
            "recall":    round(b_rec, 4),
            "f1":        round(b_f1, 4),
            "report":    b_report,
        },
        "context_aware": {
            "accuracy":  round(c_acc, 4),
            "precision": round(c_prec, 4),
            "recall":    round(c_rec, 4),
            "f1":        round(c_f1, 4),
            "report":    c_report,
        },
        "samples": {
            "train":    len(X_train),
            "test":     len(X_test),
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
