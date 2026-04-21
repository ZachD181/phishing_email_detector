import streamlit as st
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "phishing_detector.pkl"

st.set_page_config(page_title="Phishing Email Detector", page_icon="📧")

st.title("📧 Phishing Email Detector")
st.caption("Built with Python, scikit-learn, and Streamlit")
st.write("Enter an email message below to classify it as phishing or safe.")
st.divider()

if not MODEL_PATH.exists():
    st.error("Model file not found. Train the model first with: python .\\src\\train.py")
    st.stop()

model = joblib.load(MODEL_PATH)

col1, col2 = st.columns(2)
with col1:
    if st.button("Try phishing example"):
        st.session_state["example"] = "Your account has been locked. Verify immediately."
with col2:
    if st.button("Try safe example"):
        st.session_state["example"] = "Can you send me the lab notes when you have time?"

default_text = st.session_state.get("example", "")
email_text = st.text_area("Email text", value=default_text, height=180, key="email_input")

if st.button("Analyze Email"):
    if not email_text.strip():
        st.warning("Please enter some email text.")
    else:
        prediction = model.predict([email_text])[0]
        probabilities = model.predict_proba([email_text])[0]
        classes = model.classes_
        confidence = dict(zip(classes, probabilities))

        if prediction == "phishing":
            st.error("🚨 Phishing detected")
        else:
            st.success("✅ Looks safe")

        st.subheader("Confidence Scores")
        phishing_score = float(confidence.get("phishing", 0.0))
        safe_score = float(confidence.get("safe", 0.0))

        st.write(f"**Phishing:** {phishing_score:.4f}")
        st.progress(phishing_score)

        st.write(f"**Safe:** {safe_score:.4f}")
        st.progress(safe_score)

        text_lower = email_text.lower()
        risky_terms = [
            "urgent", "verify", "password", "account", "click",
            "login", "credentials", "suspend", "locked", "confirm"
        ]
        hits = [w for w in risky_terms if w in text_lower]

        st.subheader("Why this result")
        if hits:
            st.write("Suspicious language detected: " + ", ".join(sorted(set(hits))))
        else:
            st.write("No obvious high-risk keywords detected.")

        st.subheader("Note")
        st.write("This is a basic educational model and should not be used as a sole security control.")