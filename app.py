import streamlit as st
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "phishing_detector.pkl"

st.set_page_config(page_title="Phishing Email Detector", page_icon="📧")

st.title("📧 Phishing Email Detector")
st.write("Enter an email message below to classify it as phishing or safe.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Train the model first with: python .\\src\\train.py")
    st.stop()

model = joblib.load(MODEL_PATH)

email_text = st.text_area("Email text", height=180)


suspicious_keywords = [
    "verify", "urgent", "password", "account", "login",
    "click", "reward", "claim", "confirm", "suspend",
    "security", "update", "credentials", "bank", "payment"
]

if st.button("Analyze Email"):
    if not email_text.strip():
        st.warning("Please enter some email text.")
    else:
        prediction = model.predict([email_text])[0]
        probabilities = model.predict_proba([email_text])[0]
        classes = model.classes_

        # 🔥 THIS LINE WAS MISSING OR BROKEN
        confidence = dict(zip(classes, probabilities))

        st.subheader("Prediction")
        st.write(prediction.capitalize())

        st.subheader("Confidence Scores")
        for label, score in confidence.items():
            st.write(f"**{label.capitalize()}**: {score:.4f}")