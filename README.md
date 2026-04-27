
# Phishing Email Detector

## 🔗 Live Demo

https://phishingemaildetector-gvz9adghkvhxos2zh5zu8c.streamlit.app/

---

## 📌 Overview

This project is a machine learning-based phishing email detector with a Streamlit web interface.
Users can paste email text and receive a prediction indicating whether the message is **phishing** or **safe**, along with a confidence score.

---

## 🧠 How It Works

The model uses a text classification pipeline:

* **TF-IDF Vectorization** to convert text into numerical features
* **Machine Learning Model** (Naive Bayes / Logistic Regression)
* **Prediction + Probability Output**

---

## 📊 Features

* Real-time phishing detection
* Confidence scoring
* Clean web interface (Streamlit)
* Deployable ML model

---

## 📁 Dataset

* Custom email dataset (`emails.csv`)
* (Optional/Planned) SMS Spam Collection dataset for improved robustness

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* Pandas
* Streamlit

---

## 📈 Future Improvements

* Larger dataset integration
* Model comparison (Naive Bayes vs Logistic Regression)
* Improved explainability (highlight suspicious words)
* API endpoint for external use

