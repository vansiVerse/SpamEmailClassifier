from flask import Flask, render_template_string, request
import os
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_FILE = "SMSSpamCollection"
MODEL_FILE = "spam_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

app = Flask(__name__)

HTML = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Spam Email Detector</title>
  <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@500;700&display=swap" rel="stylesheet">
  <style>
    body {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      margin: 0;
      font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .container {
      background: rgba(255,255,255,0.90);
      padding: 3.2rem 2.7rem 2.7rem 2.7rem;
      border-radius: 32px;
      box-shadow: 0 14px 48px 0 rgba(102, 126, 234, 0.18), 0 2px 12px 0 rgba(118, 75, 162, 0.10);
      text-align: center;
      min-width: 350px;
      max-width: 430px;
      width: 100%;
      border: 2px solid #667eea;
      transition: box-shadow 0.3s, border 0.3s;
      backdrop-filter: blur(3px);
      position: relative;
      overflow: hidden;
    }
    .container::before {
      content: '';
      position: absolute;
      top: -60px; left: -60px;
      width: 120px; height: 120px;
      background: radial-gradient(circle, #667eea55 60%, transparent 100%);
      z-index: 0;
    }
    .container::after {
      content: '';
      position: absolute;
      bottom: -60px; right: -60px;
      width: 120px; height: 120px;
      background: radial-gradient(circle, #764ba255 60%, transparent 100%);
      z-index: 0;
    }
    .container > * { position: relative; z-index: 1; }
    h2 {
      margin-bottom: 2.2rem;
      color: #667eea;
      letter-spacing: 1.5px;
      font-weight: 700;
      font-size: 2.2rem;
    }
    form {
      margin-bottom: 0;
      margin-top: 0.5rem;
    }
    input[type=text] {
      width: 90%;
      padding: 1.1rem;
      border: 2px solid #764ba2;
      border-radius: 12px;
      font-size: 1.13rem;
      margin-bottom: 1.7rem;
      outline: none;
      transition: border 0.2s;
      background: #f4f6fb;
      margin-top: 0.5rem;
    }
    input[type=text]:focus {
      border: 2.5px solid #667eea;
      background: #e9e6f7;
    }
    .btn {
      background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
      color: #fff;
      border: none;
      border-radius: 12px;
      padding: 1rem 2.7rem;
      font-size: 1.13rem;
      cursor: pointer;
      margin: 0.7rem 0.7rem 1.5rem 0.7rem;
      box-shadow: 0 2px 12px rgba(102, 126, 234, 0.13);
      font-weight: 600;
      letter-spacing: 0.5px;
      transition: background 0.2s, transform 0.2s;
      margin-top: 0.5rem;
    }
    .btn:hover {
      background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
      transform: translateY(-2px) scale(1.04);
    }
    .result {
      font-size: 1.28rem;
      font-weight: bold;
      margin-top: 1.7rem;
      padding: 1.1rem 0;
      border-radius: 12px;
      box-shadow: 0 2px 12px rgba(102, 126, 234, 0.10);
      letter-spacing: 0.5px;
      display: inline-block;
      min-width: 170px;
    }
    .spam {
      background: linear-gradient(90deg, #ff5858 0%, #ffb199 100%);
      color: #fff;
      border: 2px solid #ff5858;
    }
    .not-spam {
      background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%);
      color: #fff;
      border: 2px solid #43e97b;
    }
  </style>
</head>
<body>
  <div class="container">
    <h2>Spam Email Detector</h2>
    <form method="post">
      <input type="text" name="message" placeholder="Type your message here..." required autocomplete="off">
      <br>
      <button class="btn" type="submit">Check</button>
      <a href="/spam-messages"><button class="btn" type="button">Show All Spam Messages</button></a>
    </form>
    {% if result is not none %}
      <div class="result {{ 'spam' if result == 'Spam' else 'not-spam' }}">Result: {{ result }}</div>
    {% endif %}
  </div>
</body>
</html>
'''

def download_dataset():
    if not os.path.exists(DATA_FILE):
        r = requests.get(DATA_URL)
        with open("smsspamcollection.zip", "wb") as f:
            f.write(r.content)
        import zipfile
        with zipfile.ZipFile("smsspamcollection.zip", "r") as zip_ref:
            zip_ref.extractall()
        os.remove("smsspamcollection.zip")

def train_and_save():
    download_dataset()
    df = pd.read_csv(DATA_FILE, sep='\t', header=None, names=['label', 'message'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label']
    vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = vectorizer.fit_transform(X)
    model = MultinomialNB()
    model.fit(X_tfidf, y)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

def load_model():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
        train_and_save()
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    return model, vectorizer

model, vectorizer = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        msg = request.form['message']
        msg_tfidf = vectorizer.transform([msg])
        pred = model.predict(msg_tfidf)[0]
        result = 'Spam' if pred == 1 else 'Not Spam'
    return render_template_string(HTML, result=result)

@app.route('/spam-messages')
def spam_messages():
    if not os.path.exists(DATA_FILE):
        download_dataset()
    df = pd.read_csv(DATA_FILE, sep='\t', header=None, names=['label', 'message'])
    spam_msgs = df[df['label'] == 'spam']['message'].tolist()
    html = '''
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>All Spam Messages</title>
      <style>
        body { background: linear-gradient(120deg, #f6d365 0%, #fda085 100%); min-height: 100vh; margin: 0; font-family: 'Segoe UI', Arial, sans-serif; display: flex; align-items: center; justify-content: center; }
        .container { background: #fff; padding: 2rem 1.5rem; border-radius: 18px; box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2); min-width: 350px; max-width: 700px; }
        h2 { color: #f76b1c; margin-bottom: 1.2rem; }
        ul { text-align: left; max-height: 400px; overflow-y: auto; padding-left: 1.2rem; }
        li { margin-bottom: 0.7rem; color: #333; }
        .btn { background: linear-gradient(90deg, #fda085 0%, #f6d365 100%); color: #fff; border: none; border-radius: 8px; padding: 0.6rem 1.5rem; font-size: 1rem; cursor: pointer; margin-top: 1.2rem; box-shadow: 0 2px 8px rgba(253, 160, 133, 0.15); transition: background 0.2s, transform 0.2s; }
        .btn:hover { background: linear-gradient(90deg, #f6d365 0%, #fda085 100%); transform: translateY(-2px) scale(1.03); }
      </style>
    </head>
    <body>
      <div class="container">
        <h2>All Spam Messages from Dataset</h2>
        <ul>
    '''
    for msg in spam_msgs:
        html += f'<li>{msg}</li>'
    html += '</ul>'
    html += '<a href="/"> <button class="btn">Back to Home</button> </a>'
    html += '</div></body></html>'
    return html

if __name__ == '__main__':
    app.run(debug=True)
