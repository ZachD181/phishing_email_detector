import json
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "phishing_detector.pkl"

model = joblib.load(MODEL_PATH)

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        email_text = body.get("email_text", "")

        if not email_text.strip():
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": json.dumps({
                    "error": "email_text is required"
                })
            }

        prediction = model.predict([email_text])[0]
        probabilities = model.predict_proba([email_text])[0]
        classes = model.classes_
        confidence = dict(zip(classes, probabilities))

        confidence = {
            label: float(score)
            for label, score in confidence.items()
        }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "prediction": prediction,
                "confidence": confidence
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "error": str(e)
            })
        }