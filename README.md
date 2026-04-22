A machine learning web application that predicts which subreddit a given text post belongs to, built with Flask and scikit-learn. Styled with a Reddit-inspired UI.

---

## 📌 Overview

This project trains and compares three supervised learning classifiers on text data scraped from various subreddits. Users can input any text and receive a real-time prediction of which subreddit it most likely belongs to, along with model confidence and performance metrics.

**Classifiable Subreddits:**
`r/AdviceForTeens` · `r/Anxiety` · `r/Colombia` · `r/GradSchool` · `r/NeutralPolitics` · `r/YouthRights` · `r/college` · `r/highschool` · `r/jobs` · `r/studentaffairs`

---

## 🧠 Models & Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Linear SVC** ⭐ | 81.4% | 0.81 | 0.81 | 0.81 |
| Logistic Regression | 80.3% | 0.80 | 0.80 | 0.80 |
| Random Forest | 78.4% | 0.78 | 0.78 | 0.78 |

> Linear SVC achieved the best overall performance and is set as the default model.

---

## 🗂️ Project Structure

```
reddit-classifier/
│
├── app.py                  # Flask backend & prediction routes
├── templates/
│   └── index.html          # Reddit-themed frontend UI
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── svd_reducer.pkl
│   ├── svd_transformer.pkl
│   ├── linear_svc.pkl
│   ├── logistic_reg.pkl
│   └── random_forest.pkl
│
├── notebook/               # (optional) Training notebook
│   └── training.ipynb
│
└── requirements.txt
```

---

## ⚙️ How It Works

1. **Text Vectorization** — Input text is transformed into a 3,000-feature TF-IDF matrix using a pre-fitted `TfidfVectorizer`.
2. **Dimensionality Reduction** — The matrix is compressed from 3,000 → 150 features using a pre-fitted `TruncatedSVD` (LSA).
3. **Classification** — The reduced feature vector is passed to the selected model, which outputs a predicted subreddit label.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/reddit-classifier.git
cd reddit-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Flask app
python app.py
```

Then open your browser and go to: **http://localhost:8080**

---

## 📦 Requirements

```
flask
flask-cors
scikit-learn==1.6.1
numpy
joblib
```

> ⚠️ **Important:** The `.pkl` model files were saved with **scikit-learn 1.6.1**. Using a different version may trigger `InconsistentVersionWarning`. To avoid this, pin your scikit-learn version to `1.6.1` as shown above.

---

## 🖥️ Features

- **Real-time prediction** — Paste any text and get an instant subreddit classification
- **Model switcher** — Toggle between Linear SVC, Logistic Regression, and Random Forest
- **Confidence bar** — Visual indicator of model accuracy
- **Leaderboard & Stats tab** — Per-class precision, recall, F1-score breakdown for all three models
- **Reddit-themed UI** — Styled to mimic the familiar Reddit interface

---

## 📊 Classification Report (Linear SVC)

| Subreddit | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| AdviceForTeens | 0.77 | 0.89 | 0.82 | 200 |
| Anxiety | 0.86 | 0.85 | 0.86 | 199 |
| Colombia | 0.74 | 0.87 | 0.80 | 124 |
| GradSchool | 0.85 | 0.80 | 0.82 | 200 |
| NeutralPolitics | 0.97 | 0.98 | 0.98 | 200 |
| YouthRights | 0.79 | 0.81 | 0.80 | 198 |
| college | 0.74 | 0.66 | 0.69 | 200 |
| highschool | 0.72 | 0.66 | 0.69 | 197 |
| jobs | 0.82 | 0.91 | 0.86 | 200 |
| studentaffairs | 0.81 | 0.69 | 0.74 | 200 |

---

## 📝 Notes

- The `models/` folder must be present in the same directory as `app.py` for the app to load correctly.
- If models fail to load, the app will still start but predictions will return a 500 error.
- The SVD reducer is a critical step in the pipeline — without it, the models will reject the 3,000-feature TF-IDF input.

---

## 📄 License

This project is for academic and educational purposes.
