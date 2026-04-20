import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "phishing_detector.pkl"

model = joblib.load(MODEL_PATH)

print("Phishing Email Detector")
print("Type an email message below.\n")

user_input = input("Email text: ")

prediction = model.predict([user_input])[0]
probabilities = model.predict_proba([user_input])[0]

classes = model.classes_
confidence = dict(zip(classes, probabilities))

print("\nPrediction:", prediction)
print("Confidence scores:")
for label, score in confidence.items():
    print(f"{label}: {score:.4f}")