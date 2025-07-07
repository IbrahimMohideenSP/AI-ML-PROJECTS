import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import DBSCAN
from transformers import pipeline

def run_classification():
    print("\n=== Classification: Predict High‑Risk COVID Patients ===")
    df = pd.read_csv("cdc_covid_data.csv", usecols=['age_group','hospitalized','death_yn'], dtype=str)
    df = df.dropna()
    df['label'] = df.apply(lambda r: 1 if r['death_yn']=="Yes" or r['hospitalized']=="Yes" else 0, axis=1)
    df['age'] = df['age_group'].str.extract('(\\d+)-').fillna(80).astype(int)
    X = df[['age']]
    y = df['label'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

def run_clustering():
    print("\n=== Clustering: Identify Infection Hotspots ===")
    df = pd.read_csv("Indiadata1.csv")
    coords = df[['latitude','longitude']]
    clustering = DBSCAN(eps=0.05, min_samples=2).fit(coords)
    df['cluster'] = clustering.labels_
    print(df)

def run_sentiment():
    print("\n=== Sentiment Analysis on COVID Tweets ===")
    df = pd.read_csv("mock_Corona_NLP.csv", encoding='latin1')
    df = df[df['Sentiment'].isin(['Positive','Negative'])]
    tweets = df['OriginalTweet'].sample(3, random_state=42).tolist()
    analyzer = pipeline("sentiment-analysis")
    for t in tweets:
        r = analyzer(t)[0]
        print(f"\nTweet: {t}\nSentiment: {r['label']} (score: {r['score']:.2f})\n")

if __name__=="__main__":
    run_classification()
    run_clustering()
    run_sentiment()
