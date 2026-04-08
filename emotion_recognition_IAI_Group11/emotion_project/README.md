# Context-Aware Emotion Recognition for Real-Time Conversational Analysis
## CSS 2203 – IAI Project | IT-C Group 11

---

## Team Members
| Roll No | Name             | Reg. No     |
|---------|------------------|-------------|
| 64      | Dev              | 240911736   |
| 23      | Taranpreet Singh | 240911294   |
| 22      | Suhani           | 240911286   |
| 32      | Keshav Krishna   | 240911352   |
| 34      | Naman Kaushik    | 240911366   |

---

## Project Structure

```
emotion_project/
├── backend/
│   ├── model.py       ← Baseline + Context-Aware model classes
│   ├── dataset.py     ← Dataset generation, training, evaluation
│   └── app.py         ← Flask REST API server
├── frontend/
│   └── index.html     ← Standalone UI (open in browser)
└── README.md
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install flask flask-cors scikit-learn numpy pandas
```

### 2. Start the backend
```bash
cd backend
python app.py
```
The API runs on **http://localhost:5000**

### 3. Open the frontend
Open `frontend/index.html` in any browser.

---

## API Endpoints

| Method | Endpoint           | Description                               |
|--------|--------------------|-------------------------------------------|
| GET    | /api/health        | Check if server is running                |
| POST   | /api/predict       | Predict emotion for a single utterance    |
| POST   | /api/conversation  | Predict across a list of utterances       |
| POST   | /api/reset         | Clear conversation context                |
| GET    | /api/train         | Train models and return metrics           |
| GET    | /api/metrics       | Return last evaluation metrics            |

### Example: POST /api/predict
```json
{ "text": "I can't believe this happened, I'm furious!" }
```

Response:
```json
{
  "text": "...",
  "baseline":      { "emotion": "anger", "confidence": 0.72, "probs": {...} },
  "context_aware": { "emotion": "anger", "confidence": 0.68, "probs": {...} }
}
```

---

## Emotion Classes
`neutral · joy · sadness · anger · fear · disgust · surprise`
(aligned with MELD and DailyDialog conventions)

---

## Models

### Baseline Model
- **Pipeline**: TF-IDF (1-2 grams, 5000 features, sublinear TF) → Logistic Regression
- **Feature**: Current utterance only
- **Limitation**: No conversational memory → abrupt emotion shifts

### Context-Aware Model
- **Pipeline**: Same as baseline but input = sliding window of last 3 utterances
- **Temporal Smoothing**: Transition cost matrix penalises emotionally implausible jumps
- **Smoothing alpha**: 0.25 (tunable)
- **Context window**: 3 turns (tunable)
- **Result**: Smoother, more realistic emotion trajectories

---

## Evaluation Metrics
- Accuracy, Precision (weighted), Recall (weighted), F1-Score (weighted)
- Classification report per emotion class
- Baseline vs Context-Aware comparison
