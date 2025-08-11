import os
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_FILE = "SMSSpamCollection"

# Download and extract dataset if not present
def download_dataset():
    if not os.path.exists(DATA_FILE):
        print("Downloading dataset...")
        r = requests.get(DATA_URL)
        with open("smsspamcollection.zip", "wb") as f:
            f.write(r.content)
        import zipfile
        with zipfile.ZipFile("smsspamcollection.zip", "r") as zip_ref:
            zip_ref.extractall()
        os.remove("smsspamcollection.zip")
        print("Dataset downloaded and extracted.")
    else:
        print("Dataset already exists.")

def load_data():
    df = pd.read_csv(DATA_FILE, sep='\t', header=None, names=['label', 'message'])
    return df

def preprocess(df):
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

def main():
    download_dataset()
    df = load_data()
    df = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    # Predict new email
    while True:
        msg = input("\nEnter an email message to classify (or 'exit' to quit): ")
        if msg.lower() == 'exit':
            break
        msg_tfidf = vectorizer.transform([msg])
        pred = model.predict(msg_tfidf)[0]
        print("Spam" if pred == 1 else "Not Spam")

if __name__ == "__main__":
    main()
