import streamlit as st
import numpy as np
from joblib import load
from huggingface_hub import hf_hub_download

# -------------------------------
# Load model from Hugging Face
# -------------------------------
MODEL_REPO = "Kavyasri-0612/language"
MODEL_FILE = "model.pkl"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE
)
model = load(model_path)

# -------------------------------
# Load vectorizer from GitHub (local file)
# -------------------------------
vectorizer = load("vectorizer.pkl")

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Language Detector", page_icon="🌍", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌍 Language Detector </h1>", unsafe_allow_html=True)
st.info("⚠️ This model supports only 22 languages. Please enter text based on those languages.")

st.markdown("### 🌐 Supported Languages")
st.write("English, Hindi, Tamil, French, Spanish, German, Italian, Portuguese, Dutch, Russian, Chinese, Japanese, Korean, etc.")

text = st.text_area("✍️ Enter your text below:", height=150)

if st.button("🔍 Detect Language"):
    if text.strip():
        transformed = vectorizer.transform([text])
        probs = model.predict_proba(transformed)
        confidence = np.max(probs)

        if confidence < 0.6:
            st.warning("⚠️ Language not recognized (may not be in dataset)")
        else:
            result = model.predict(transformed)[0]
            st.success(f"🌐 Detected Language: **{result}**")
            st.progress(int(confidence * 100))
            st.write(f"📊 Prediction Reliability: {confidence:.2f}")
    else:
        st.warning("Please enter some text")

st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with ❤️ using Machine Learning & NLP</p>", unsafe_allow_html=True)