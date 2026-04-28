import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "emails.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "phishing_detector.pkl"

df = pd.read_csv(DATA_PATH)
df = df.dropna()
df["label"] = df["label"].str.strip().str.lower()
df["text"] = df["text"].astype(str).str.strip().str.lower()

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", MultinomialNB())
    ])
}

best_model = None
best_name = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\n{name}")
    print("-" * len(name))
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print(classification_report(y_test, predictions))

    if accuracy > best_score:
        best_score = accuracy
        best_name = name
        best_model = model

MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(best_model, MODEL_PATH)

print(f"\nBest model: {best_name}")
print(f"Best accuracy: {best_score:.4f}")
print(f"Saved best model to: {MODEL_PATH}")