
# AI-Powered Healthcare & COVID-19 Analysis
# Includes Classification (Random Forest), Clustering (DBSCAN), and Sentiment Analysis (BERT)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import pipeline

# === 1. Classification using Random Forest ===
# Simulated patient data
df = pd.DataFrame({
    'age': [25, 65, 45, 33, 29, 62],
    'fever': [1, 0, 1, 1, 0, 1],
    'cough': [1, 1, 0, 1, 1, 0],
    'oxygen': [96, 88, 93, 97, 95, 87],
    'label': [0, 1, 1, 0, 0, 1]  # 1 = high-risk, 0 = low-risk
})

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# === 2. Clustering of infection hotspots using DBSCAN ===
coords = pd.DataFrame({
    'latitude': [13.08, 13.07, 12.99, 13.01, 13.45],
    'longitude': [80.27, 80.28, 80.25, 80.20, 80.22]
})

db = DBSCAN(eps=0.05, min_samples=2).fit(coords)
coords['cluster'] = db.labels_

print("\n=== DBSCAN Clustering Results ===")
print(coords)

# === 3. Sentiment Analysis using BERT ===
analyzer = pipeline("sentiment-analysis")
tweets = [
    "Vaccines are saving lives!",
    "This lockdown is terrible!",
    "I’m worried about the rising cases."
]

print("\n=== Sentiment Analysis Results ===")
for tweet in tweets:
    result = analyzer(tweet)
    print(f"Tweet: {tweet}\nSentiment: {result[0]['label']} (Score: {result[0]['score']:.2f})\n")
