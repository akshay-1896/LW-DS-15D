# Emotion Signal

A Streamlit UI for the emotion classifier from `Emotions_NLP.ipynb` — TF-IDF +
Logistic Regression trained on `train.txt` (16,000 labeled sentences, 6 emotions:
sadness, anger, love, surprise, fear, joy).

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep `app.py` and `train.txt` in the same folder — the app trains the model
once on startup (a couple of seconds) and caches it for the rest of the session.

## What it does

- Cleans input text the same way the notebook does: lowercase, strip
  punctuation, strip digits, strip non-ASCII characters, drop stopwords.
- Vectorizes with TF-IDF and predicts with Logistic Regression (the
  highest-accuracy model in the notebook).
- Shows the predicted **label number** (0–5) front and center, plus the
  emotion name and a confidence breakdown across all six classes.

## Note on stopwords

The notebook uses NLTK's stopword list, which requires a one-time internet
download (`nltk.download(...)`). This app uses scikit-learn's built-in
English stopword list instead — it's bundled with scikit-learn, needs no
download, and keeps the app fast to start. The rest of the cleaning pipeline
matches the notebook exactly. If you'd rather match NLTK's list precisely,
swap `ENGLISH_STOP_WORDS` in `app.py` for `nltk.corpus.stopwords.words('english')`.

## Label mapping

Labels are assigned by order of first appearance in `train.txt`, exactly as
in the notebook:

| Label | Emotion  |
|-------|----------|
| 0     | sadness  |
| 1     | anger    |
| 2     | love     |
| 3     | surprise |
| 4     | fear     |
| 5     | joy      |
