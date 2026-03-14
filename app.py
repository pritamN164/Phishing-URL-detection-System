import streamlit as st
import numpy as np
import re
import joblib
import pandas as pd

from scipy.sparse import hstack, csr_matrix

# Load Model and Metrics
model = joblib.load("phishing_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
metrics = joblib.load("model_metrics.pkl")

# Feature Extraction Function
def extract_features(url):
    features = []

    features.append(len(url))
    features.append(url.count('.'))
    features.append(url.count('-'))
    features.append(url.count('@'))
    features.append(url.count('?'))
    features.append(url.count('='))
    features.append(url.count('http'))
    features.append(url.count('https'))
    features.append(url.count('www'))
    features.append(url.count('/'))

    features.append(sum(c.isdigit() for c in url))

    suspicious_words = ['login','verify','bank','secure','account','update']

    count = 0
    for word in suspicious_words:
        if word in url.lower():
            count += 1

    features.append(count)

    ip_pattern = r'(\d{1,3}\.){3}\d{1,3}'

    if re.search(ip_pattern,url):
        features.append(1)
    else:
        features.append(0)

    return features

# Streamlit UI
st.title("🔐 Phishing URL Detection System")
st.write("Machine Learning based system to detect **Phishing URLs** using URL features and NLP techniques.")

# Sidebar Information
st.sidebar.title("📊 Model Information")

st.sidebar.write("Model Used: Random Forest")

st.sidebar.write("Dataset Size Used:")
st.sidebar.write(metrics["dataset_size"])

st.sidebar.write("Model Accuracy:")
st.sidebar.write(round(metrics["accuracy"]*100,2), "%")

# URL Prediction Section
st.header("🔍 Check URL")

url = st.text_input("Enter a URL to check")

if st.button("Analyze URL"):

    if url == "":
        st.warning("Please enter a URL")

    else:

        url = url.lower().strip()

        # Trusted domain list
        trusted_sites = [
            "google.com",
            "youtube.com",
            "facebook.com",
            "amazon.in",
            "sbi.bank.in",
            "instagram.com"
        ]

        # Check trusted domains first
        for site in trusted_sites:
            if site in url:
                st.success("✅ Trusted Website (Known Domain)")
                st.stop()

        # ML Feature Extraction
        ml_features = csr_matrix([extract_features(url)])

        # NLP Feature Extraction
        nlp_features = tfidf.transform([url])

        # Combine Features
        X = hstack([ml_features, nlp_features])

        # Prediction
        prediction = model.predict(X)

        if prediction[0] == 1:
            st.error("⚠️ Phishing URL Detected")
        else:
            st.success("✅ Legitimate URL")

# Model Performance Section
st.header("📈 Model Performance")

st.subheader("Accuracy")
st.write(round(metrics["accuracy"]*100,2), "%")


st.subheader("Confusion Matrix")

conf_matrix = pd.DataFrame(
    metrics["confusion_matrix"],
    columns=["Predicted Legit","Predicted Phishing"],
    index=["Actual Legit","Actual Phishing"]
)

st.table(conf_matrix)


st.subheader("Classification Report")
st.text(metrics["classification_report"])

# Project Description
st.header("📘 Project Description")

st.write("""
This project detects phishing websites using Machine Learning and Natural Language Processing techniques.

Features used include:

• URL structural features (length, dots, symbols)  
• Suspicious keyword detection  
• Character n-gram TF-IDF features  
• Random Forest classification model

The model analyzes patterns in URLs to determine whether they are **legitimate or phishing**.
""")