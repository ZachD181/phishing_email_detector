import streamlit as st
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "phishing_detector.pkl"

st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Phishing Email Detector")
st.caption("Machine learning app built with Python, scikit-learn, and Streamlit.")

st.info(
    "Paste an email message below and the model will classify it as "
    "**phishing** or **safe** with confidence scores."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Train the model first with: `python src/train.py`")
    st.stop()

model = joblib.load(MODEL_PATH)

st.subheader("Try an example")

col1, col2 = st.columns(2)

with col1:
    if st.button("🚨 Phishing example"):
        st.session_state["example"] = (
            "Your account has been locked. Verify your password immediately "
            "by clicking this link."
        )

with col2:
    if st.button("✅ Safe example"):
        st.session_state["example"] = (
            "Can you send me the lab notes when you have time?"
        )

email_text = st.text_area(
    "Email text",
    value=st.session_state.get("example", ""),
    height=180,
    placeholder="Paste an email message here..."
)

analyze = st.button("Analyze Email", type="primary")

if analyze:
    if not email_text.strip():
        st.warning("Please enter some email text first.")
    else:
        prediction = model.predict([email_text])[0]
        probabilities = model.predict_proba([email_text])[0]
        classes = model.classes_
        confidence = dict(zip(classes, probabilities))

        phishing_score = float(confidence.get("phishing", 0.0))
        safe_score = float(confidence.get("safe", 0.0))

        st.divider()

        st.subheader("Result")

        if prediction == "phishing":
            st.error(f"🚨 Phishing detected — {phishing_score * 100:.2f}% confidence")
        else:
            st.success(f"✅ Looks safe — {safe_score * 100:.2f}% confidence")

        st.subheader("Confidence Scores")

        st.write(f"**Phishing:** {phishing_score * 100:.2f}%")
        st.progress(phishing_score)

        st.write(f"**Safe:** {safe_score * 100:.2f}%")
        st.progress(safe_score)

        text_lower = email_text.lower()
        risky_terms = [
            "urgent", "verify", "password", "account", "click",
            "login", "credentials", "suspend", "locked", "confirm"
        ]
        hits = [word for word in risky_terms if word in text_lower]

        st.subheader("Why this result?")

        if hits:
            st.warning(
                "Suspicious language detected: "
                + ", ".join(sorted(set(hits)))
            )
        else:
            st.write("No obvious high-risk keywords detected.")

st.divider()

st.caption(
    "Note: This is an educational machine learning project and should not be "
    "used as the only security tool for evaluating real emails."
)