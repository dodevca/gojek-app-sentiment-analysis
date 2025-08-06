from flask import Flask, request, render_template
import joblib
import numpy as np
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)

model             = joblib.load("model/model.joblib")
tfidf_transformer = joblib.load("model/vectorizer.joblib")

stemmer = StemmerFactory().create_stemmer()

INDONESIAN_STOPWORDS = set([
    'yang', 'dari', 'dan', 'di', 'ke', 'untuk', 'pada', 'adalah', 'itu', 'ini', 'dengan',
    'sebagai', 'oleh', 'juga', 'karena', 'atau', 'sudah', 'belum', 'akan', 'bisa', 'dalam',
    'tidak', 'tersebut', 'lebih', 'maupun', 'apa', 'agar', 'kami', 'kita', 'saya', 'mereka',
    'anda', 'aku', 'kau', 'kamu', 'nya', 'lah', 'pun', 'hanya', 'jadi', 'masih', 'masuk',
    'keluar', 'tanpa', 'semua', 'antara', 'setelah', 'sebelum', 'hingga', 'sejak', 'bagai',
    'bagi', 'yakni', 'bahwa', 'oleh', 'begitu', 'jika', 'namun', 'walaupun', 'maka', 'harus',
    'bukan', 'boleh', 'perlu', 'telah', 'dapat', 'tentang', 'menjadi', 'hingga', 'sedang',
    'lagi', 'mau', 'sudah', 'tapi', 'karena', 'seperti', 'bahkan', 'supaya', 'biar', 'punya',
    'cukup', 'selalu', 'baru', 'lama', 'id', 'yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo',
    'kalo', 'smp', 'biar', 'bikin', 'bilang', 'gak', 'ga', 'udh', 'udah', 'gini', 'gitu',
    'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 'pun', 'siap', 'gua', 'gue', 'aq', 'aku',
    'gt', 'ttp', 'mo', 'kpd', 'idk', 'deh', 'btw', 'abng', 'utk', 'sbh', 'lho', 'nder', 'sbg',
    'be', 'like', 'tidk', 'gw', 'brng', 'hrs', 'jwb', 'hoy', 'tp', 'shrsny', 'sy', 'klu',
    'as', 'ae', 'pekok', 'ah', 'gaessh', 'guys', 'ges', 'gaes', 'ap', 'aj', 'ih'
])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [t for t in text.split() if t not in INDONESIAN_STOPWORDS]
    stems  = [stemmer.stem(t) for t in tokens]

    return ' '.join(stems)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction    = None
    probabilities = None
    sorted_probs  = None
    raw_text      = ""

    if request.method == "POST":
        raw_text      = request.form["text"]
        text          = preprocess_text(raw_text)
        X_feat        = tfidf_transformer.transform([text])
        proba         = model.predict_proba(X_feat)[0]
        classes       = model.classes_
        idx_max       = np.argmax(proba)
        prediction    = classes[idx_max]
        probabilities = {cls: f"{p*100:.2f}%" for cls, p in zip(classes, proba)}

        sorted_probs = sorted(
            probabilities.items(),
            key=lambda x: float(x[1].rstrip('%')),
            reverse=True
        )

    return render_template("index.html", prediction=prediction, probabilities=sorted_probs, text=raw_text)

if __name__ == "__main__":
    app.run(debug=True)