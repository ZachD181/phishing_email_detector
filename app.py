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

if st.button("Analyze Email"):
    if not email_text.strip():
        st.warning("Please enter some email text.")
    else:
        prediction = model.predict([email_text])[0]
        probabilities = model.predict_proba([email_text])[0]
        classes = model.classes_
        confidence = dict(zip(classes, probabilities))

        phishing_score = confidence.get("phishing", 0)
        safe_score = confidence.get("safe", 0)

        st.subheader("Prediction")

        if prediction == "phishing":
            st.error("This email appears to be phishing.")
        else:
            st.success("This email appears to be safe.")

        st.subheader("Confidence Scores")
        st.write(f"**Phishing:** {phishing_score:.4f}")
        st.write(f"**Safe:** {safe_score:.4f}")

        st.progress(float(phishing_score))

        st.subheader("Quick Interpretation")
        if phishing_score > 0.80:
            st.write("High phishing likelihood.")
        elif phishing_score > 0.60:
            st.write("Moderate phishing likelihood.")
        else:
            st.write("Low phishing likelihood.")
        prediction = model.predict([email_text])[0]
        probabilities = model.predict_proba([email_text])[0]
        classes = model.classes_
        confidence = dict(zip(classes, probabilities))

        st.subheader("Prediction")
        st.write(prediction.capitalize())

        st.subheader("Confidence Scores")
        for label, score in confidence.items():
            st.write(f"**{label.capitalize()}**: {score:.4f}")