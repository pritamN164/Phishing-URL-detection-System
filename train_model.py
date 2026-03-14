import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from scipy.sparse import hstack, csr_matrix

# Load Dataset
data = pd.read_csv(r"C:\Users\prita\OneDrive\Desktop\bml casestudy\malicious_phish.csv")

# Keep only phishing and benign
data = data[data['type'].isin(['phishing','benign'])]

data['label'] = data['type'].map({
    'phishing':1,
    'benign':0
})
# Reduce dataset size (memory friendly)
data = data.sample(n=40000, random_state=42)


urls = data['url'].astype(str)
labels = data['label']

# Feature Extraction
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

    # digit count
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

# Create ML Features
ml_features = csr_matrix([extract_features(url) for url in urls])

# NLP Features
tfidf = TfidfVectorizer(
    max_features=1500,
    analyzer='char',
    ngram_range=(3,5),
    lowercase=True
)
nlp_features = tfidf.fit_transform(urls)

# Combine Features
X = hstack([ml_features,nlp_features])
y = labels

# Train Test Split
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=120,
    max_depth=18,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train,y_train)

# Evaluate Model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

conf_matrix = confusion_matrix(y_test,y_pred)

class_report = classification_report(y_test,y_pred)


print("\nModel Accuracy:",accuracy)
print("\nConfusion Matrix:\n",conf_matrix)
print("\nClassification Report:\n",class_report)

# Save Model and Metrics
joblib.dump(model,"phishing_model.pkl")
joblib.dump(tfidf,"tfidf_vectorizer.pkl")

metrics = {
    "accuracy": accuracy,
    "confusion_matrix": conf_matrix,
    "classification_report": class_report,
    "dataset_size": len(data)
}

joblib.dump(metrics,"model_metrics.pkl")

print("\nModel, vectorizer and metrics saved successfully!")