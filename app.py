import streamlit as st
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "phishing_detector.pkl"

st.set_page_config(page_title="Phishing Email Detector", page_icon="📧")

st.caption("Built with Python, scikit-learn, and Streamlit")
st.divider()

st.title("📧 Phishing Email Detector")
st.write("Enter an email message below to classify it as phishing or safe.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Train the model first with: python .\\src\\train.py")
    st.stop()

model = joblib.load(MODEL_PATH)



suspicious_keywords = [
    "verify", "urgent", "password", "account", "login",
    "click", "reward", "claim", "confirm", "suspend",
    "security", "update", "credentials", "bank", "payment"
]


col1, col2 = st.columns(2)
with col1:
    if st.button("Try phishing example"):
        st.session_state["example"] = "Your account has been locked. Verify immediately."
with col2:
    if st.button("Try safe example"):
        st.session_state["example"] = "Can you send me the lab notes when you have time?"

default_text = st.session_state.get("example", "")
email_text = st.text_area("Email text", value=default_text, height=180)

if st.button("Analyze Email"):
    if not email_text.strip():
        st.warning("Please enter some email text.")
    else:
        prediction = model.predict([email_text])[0]
        probabilities = model.predict_proba([email_text])[0]
        classes = model.classes_
        confidence = dict(zip(classes, probabilities))

        # Result banner
        if prediction == "phishing":
            st.error("🚨 Phishing detected")
        else:
            st.success("✅ Looks safe")

        # Confidence
        st.subheader("Confidence Scores")
        for label, score in confidence.items():
            st.write(f"**{label.capitalize()}**: {score:.4f}")

        # Simple explanation (keyword-based)
        text_lower = email_text.lower()
        risky_terms = [
            "urgent", "verify", "password", "account", "click",
            "login", "credentials", "suspend", "locked", "confirm"
        ]
        hits = [w for w in risky_terms if w in text_lower]

        st.subheader("Why this result")
        if hits:
            st.write("Suspicious language detected:", ", ".join(sorted(set(hits))))
        else:
            st.write("No obvious high-risk keywords detected.")