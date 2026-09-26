"""
Emotion Signal — a text emotion classifier
Trained inline (TF-IDF + Logistic Regression) on the same pipeline used in
Emotions_NLP.ipynb, wrapped in a Streamlit interface.
"""

import os
import re
import string

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Emotion Signal",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Design tokens — six emotions get six accent colors, everything else stays quiet
# --------------------------------------------------------------------------
COLORS = {
    "bg": "#1C1A26",
    "surface": "#252230",
    "surface_alt": "#2F2B3D",
    "text": "#F3EFE7",
    "muted": "#A9A3B8",
    "border": "#3A3649",
}

EMOTION_META = {
    "sadness":  {"color": "#5B84C4", "glyph": "◐"},
    "anger":    {"color": "#D9483A", "glyph": "◆"},
    "love":     {"color": "#E8749A", "glyph": "♥"},
    "surprise": {"color": "#4FC1C7", "glyph": "✦"},
    "fear":     {"color": "#8C6FD1", "glyph": "▲"},
    "joy":      {"color": "#F2B84B", "glyph": "●"},
}

EXAMPLES = [
    "i finally got the promotion i have been working towards all year",
    "i cannot believe they cancelled the trip without even telling me",
    "the empty side of the bed still doesnt feel normal",
    "the thunder cracked so close it rattled every window in the house",
    "he wrote her a letter every single day she was away",
    "i did not expect the whole team to show up at my door",
]

# --------------------------------------------------------------------------
# Global CSS
# --------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: {COLORS['bg']};
    color: {COLORS['text']};
}}

section[data-testid="stSidebar"] {{
    background: {COLORS['surface']};
    border-right: 1px solid {COLORS['border']};
}}

h1, h2, h3, .display-font {{
    font-family: 'Fraunces', serif;
}}

.mono {{
    font-family: 'IBM Plex Mono', monospace;
}}

/* Hero */
.hero-eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-size: 0.72rem;
    color: {COLORS['muted']};
    margin-bottom: 0.3rem;
}}
.hero-title {{
    font-family: 'Fraunces', serif;
    font-size: 2.6rem;
    font-weight: 600;
    line-height: 1.15;
    margin: 0 0 0.5rem 0;
}}
.hero-sub {{
    color: {COLORS['muted']};
    font-size: 1rem;
    max-width: 640px;
    line-height: 1.55;
}}

/* Panels */
.panel {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 14px;
    padding: 1.6rem 1.7rem;
}}

textarea {{
    font-family: 'Inter', sans-serif !important;
    background: {COLORS['surface_alt']} !important;
    color: {COLORS['text']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 10px !important;
}}

button[kind="primary"] {{
    background: {COLORS['text']} !important;
    color: {COLORS['bg']} !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-size: 0.78rem !important;
    padding: 0.6rem 1.3rem !important;
    transition: transform 0.15s ease, opacity 0.15s ease;
}}
button[kind="primary"]:hover {{
    transform: translateY(-1px);
    opacity: 0.88;
    color: {COLORS['bg']} !important;
}}

button[kind="secondary"] {{
    background: {COLORS['surface_alt']} !important;
    color: {COLORS['muted']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 999px !important;
    font-family: 'Inter', sans-serif !important;
    text-transform: none !important;
    font-weight: 400 !important;
    font-size: 0.8rem !important;
    padding: 0.35rem 0.9rem !important;
}}
button[kind="secondary"]:hover {{
    border-color: {COLORS['muted']} !important;
    color: {COLORS['text']} !important;
}}

/* Result hero: the label number */
.result-wrap {{
    display: flex;
    align-items: center;
    gap: 1.4rem;
    margin-bottom: 1.6rem;
}}
.label-badge {{
    width: 108px;
    height: 108px;
    min-width: 108px;
    border-radius: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Fraunces', serif;
    font-size: 3.2rem;
    font-weight: 600;
    color: {COLORS['bg']};
}}
.result-emotion {{
    font-family: 'Fraunces', serif;
    font-size: 1.8rem;
    text-transform: capitalize;
    margin: 0;
}}
.result-caption {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {COLORS['muted']};
    margin: 0.15rem 0 0 0;
}}

/* Spectrum meter */
.meter-row {{
    display: grid;
    grid-template-columns: 100px 1fr 56px;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.55rem;
}}
.meter-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: {COLORS['muted']};
    text-transform: capitalize;
}}
.meter-track {{
    background: {COLORS['surface_alt']};
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
    border: 1px solid {COLORS['border']};
}}
.meter-fill {{
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s cubic-bezier(0.22, 1, 0.36, 1);
}}
.meter-pct {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: {COLORS['muted']};
    text-align: right;
}}

.footer-note {{
    color: {COLORS['muted']};
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 2rem;
    border-top: 1px solid {COLORS['border']};
    padding-top: 1rem;
}}

hr {{ border-color: {COLORS['border']}; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Preprocessing — mirrors the notebook's cleaning steps exactly (lowercase,
# strip punctuation, strip digits, strip non-ASCII / emoji, drop stopwords).
# sklearn's built-in stopword list is used in place of NLTK's so the app
# needs no separate corpus download and stays fast to start.
# --------------------------------------------------------------------------
STOPWORDS = ENGLISH_STOP_WORDS


def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = "".join(ch for ch in text if not ch.isdigit())
    text = "".join(ch for ch in text if ch.isascii())
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)


# --------------------------------------------------------------------------
# Train once, cache for the life of the app
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_and_train():
    data_path = os.path.join(os.path.dirname(__file__), "train.txt")
    df = pd.read_csv(data_path, sep=";", header=None, names=["text", "emotion"])

    unique_emotions = df["emotion"].unique()
    emotion_to_id = {e: i for i, e in enumerate(unique_emotions)}
    id_to_emotion = {i: e for e, i in emotion_to_id.items()}
    df["label"] = df["emotion"].map(emotion_to_id)

    df["clean_text"] = df["text"].apply(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_vec))
    class_counts = df["emotion"].value_counts()

    return {
        "model": model,
        "vectorizer": vectorizer,
        "id_to_emotion": id_to_emotion,
        "accuracy": accuracy,
        "n_rows": len(df),
        "class_counts": class_counts,
    }


def predict(text, artifacts):
    clean = preprocess(text)
    if not clean.strip():
        return None
    vec = artifacts["vectorizer"].transform([clean])
    proba = artifacts["model"].predict_proba(vec)[0]
    label = int(np.argmax(proba))
    return {
        "label": label,
        "emotion": artifacts["id_to_emotion"][label],
        "proba": proba,
        "clean": clean,
    }


with st.spinner("Training the model on 16,000 labeled sentences…"):
    artifacts = load_and_train()

if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "result" not in st.session_state:
    st.session_state.result = None

# --------------------------------------------------------------------------
# Sidebar — model card
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="hero-eyebrow">Model Card</p>', unsafe_allow_html=True)
    st.markdown('<p class="display-font" style="font-size:1.3rem; margin-top:-0.3rem;">TF-IDF + Logistic Regression</p>', unsafe_allow_html=True)

    st.metric("Held-out accuracy", f"{artifacts['accuracy']*100:.1f}%")
    st.metric("Training sentences", f"{artifacts['n_rows']:,}")

    st.markdown("---")
    st.markdown('<p class="hero-eyebrow">Label Key</p>', unsafe_allow_html=True)
    for i in range(6):
        emo = artifacts["id_to_emotion"][i]
        meta = EMOTION_META[emo]
        st.markdown(
            f"""<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.45rem;">
                <div style="width:10px; height:10px; border-radius:3px; background:{meta['color']};"></div>
                <span class="mono" style="font-size:0.82rem; color:{COLORS['text']};">{i}</span>
                <span style="font-size:0.85rem; color:{COLORS['muted']}; text-transform:capitalize;">{emo}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<p class="hero-eyebrow">Class Balance</p>', unsafe_allow_html=True)
    counts = artifacts["class_counts"]
    max_count = counts.max()
    for emo, count in counts.items():
        meta = EMOTION_META[emo]
        pct = count / max_count * 100
        st.markdown(
            f"""<div style="margin-bottom:0.5rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:{COLORS['muted']}; margin-bottom:2px;">
                    <span style="text-transform:capitalize;">{emo}</span><span class="mono">{count}</span>
                </div>
                <div style="background:{COLORS['surface_alt']}; border-radius:5px; height:6px; overflow:hidden;">
                    <div style="background:{meta['color']}; width:{pct}%; height:100%;"></div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown('<p class="hero-eyebrow">Text Emotion Classifier</p>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">What is this sentence feeling?</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Type a sentence and the model reads it the way it read '
    '16,000 journal-style lines during training — lowercased, stripped of '
    'punctuation, numbers and stopwords — then scores it against six emotions: '
    'sadness, anger, love, surprise, fear and joy.</p>',
    unsafe_allow_html=True,
)
st.write("")

# --------------------------------------------------------------------------
# Input panel
# --------------------------------------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    text_input = st.text_area(
        "Sentence",
        value=st.session_state.input_text,
        height=140,
        placeholder="e.g. i cannot stop smiling since i heard the news",
        label_visibility="collapsed",
    )

    st.markdown('<p class="hero-eyebrow" style="margin-top:0.9rem;">Try one</p>', unsafe_allow_html=True)
    chip_cols = st.columns(3)
    for i, example in enumerate(EXAMPLES):
        with chip_cols[i % 3]:
            if st.button(example[:24] + "…", key=f"ex_{i}", use_container_width=True):
                st.session_state.input_text = example
                st.rerun()

    st.write("")
    analyze = st.button("Analyze emotion", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

if analyze:
    st.session_state.input_text = text_input
    result = predict(text_input, artifacts)
    st.session_state.result = result
    if result is None:
        st.session_state.result = None

with right:
    st.markdown('<div class="panel" style="min-height:100%;">', unsafe_allow_html=True)

    if analyze and st.session_state.result is None:
        st.markdown('<p class="hero-eyebrow">Result</p>', unsafe_allow_html=True)
        st.warning("That sentence had nothing left after cleanup (only stopwords/punctuation). Try a fuller sentence.")
        st.markdown('</div>', unsafe_allow_html=True)
    elif st.session_state.result is not None:
        result = st.session_state.result
        meta = EMOTION_META[result["emotion"]]

        st.markdown('<p class="hero-eyebrow">Result</p>', unsafe_allow_html=True)
        st.markdown(
            f"""<div class="result-wrap">
                <div class="label-badge" style="background:{meta['color']};">{result['label']}</div>
                <div>
                    <p class="result-emotion">{meta['glyph']} {result['emotion']}</p>
                    <p class="result-caption">predicted label · {result['proba'][result['label']]*100:.1f}% confidence</p>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        order = np.argsort(result["proba"])[::-1]
        for idx in order:
            emo = artifacts["id_to_emotion"][idx]
            m = EMOTION_META[emo]
            pct = result["proba"][idx] * 100
            st.markdown(
                f"""<div class="meter-row">
                    <span class="meter-label">{emo}</span>
                    <div class="meter-track">
                        <div class="meter-fill" style="width:{pct:.1f}%; background:{m['color']};"></div>
                    </div>
                    <span class="meter-pct">{pct:.1f}%</span>
                </div>""",
                unsafe_allow_html=True,
            )

        with st.expander("What the model actually saw"):
            st.markdown(
                f'<span class="mono" style="color:{COLORS["muted"]};">'
                f'"{result["clean"] or "(nothing left after cleanup)"}"</span>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="hero-eyebrow">Result</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:{COLORS["muted"]}; font-size:0.9rem;">'
            f'Nothing analyzed yet — write a sentence on the left and press '
            f'<span class="mono">Analyze emotion</span>.</p>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<p class="footer-note">TF-IDF vectorizer + Logistic Regression, trained fresh on app start · '
    'dataset: train.txt (16,000 labeled sentences, 6 emotions)</p>',
    unsafe_allow_html=True,
)
