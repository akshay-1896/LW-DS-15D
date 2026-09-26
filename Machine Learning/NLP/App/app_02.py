"""
Emotion Signal — a text emotion classifier
Trained inline (TF-IDF + Logistic Regression) on the same pipeline used in
Emotions_NLP.ipynb, wrapped in a Streamlit interface with an abstract
gradient "aurora" visual theme.
"""

import os
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
# Design tokens — each emotion carries a two-stop gradient rather than a flat
# color; those six gradients also seed the ambient aurora background.
# --------------------------------------------------------------------------
COLORS = {
    "bg": "#0A0A16",
    "text": "#F5F3FF",
    "muted": "#ACA7CC",
    "glass": "rgba(255,255,255,0.055)",
    "glass_alt": "rgba(255,255,255,0.09)",
    "border": "rgba(255,255,255,0.14)",
}

EMOTION_META = {
    "sadness":  {"c1": "#4F6FE0", "c2": "#8FB4FF", "glyph": "◐"},
    "anger":    {"c1": "#FF5F6D", "c2": "#FFA26B", "glyph": "◆"},
    "love":     {"c1": "#FF6FA5", "c2": "#FFB0CE", "glyph": "♥"},
    "surprise": {"c1": "#19C3E6", "c2": "#7DE8E0", "glyph": "✦"},
    "fear":     {"c1": "#7C5CFF", "c2": "#B79BFF", "glyph": "▲"},
    "joy":      {"c1": "#FFA928", "c2": "#FFE066", "glyph": "●"},
}


def grad(emo, angle=135):
    m = EMOTION_META[emo]
    return f"linear-gradient({angle}deg, {m['c1']}, {m['c2']})"


EXAMPLES = [
    "i finally got the promotion i have been working towards all year",
    "i cannot believe they cancelled the trip without even telling me",
    "the empty side of the bed still doesnt feel normal",
    "the thunder cracked so close it rattled every window in the house",
    "he wrote her a letter every single day she was away",
    "i did not expect the whole team to show up at my door",
]

# --------------------------------------------------------------------------
# Global CSS — glassmorphism panels floating over an animated gradient
# "aurora" made from blurred blobs in the six emotion colors.
# --------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

html, body {{ background: {COLORS['bg']}; }}

.stApp {{
    background: transparent;
    color: {COLORS['text']};
}}

/* ---- Ambient aurora background ---- */
.aurora-bg {{
    position: fixed;
    inset: 0;
    z-index: -1;
    overflow: hidden;
    background: {COLORS['bg']};
}}
.blob {{
    position: absolute;
    border-radius: 50%;
    filter: blur(85px);
    opacity: 0.5;
    mix-blend-mode: screen;
    will-change: transform;
}}
.blob-sadness  {{ width: 460px; height: 460px; top: -12%; left: -6%;
    background: radial-gradient(circle, {EMOTION_META['sadness']['c1']}, transparent 70%);
    animation: floatA 24s ease-in-out infinite; }}
.blob-anger    {{ width: 380px; height: 380px; top: 55%; left: -10%;
    background: radial-gradient(circle, {EMOTION_META['anger']['c1']}, transparent 70%);
    animation: floatB 20s ease-in-out infinite; }}
.blob-love     {{ width: 420px; height: 420px; top: -8%; right: -8%;
    background: radial-gradient(circle, {EMOTION_META['love']['c1']}, transparent 70%);
    animation: floatC 26s ease-in-out infinite; }}
.blob-surprise {{ width: 340px; height: 340px; bottom: -12%; right: 10%;
    background: radial-gradient(circle, {EMOTION_META['surprise']['c1']}, transparent 70%);
    animation: floatA 18s ease-in-out infinite reverse; }}
.blob-fear     {{ width: 400px; height: 400px; top: 30%; left: 35%;
    background: radial-gradient(circle, {EMOTION_META['fear']['c1']}, transparent 70%);
    animation: floatB 28s ease-in-out infinite reverse; }}
.blob-joy      {{ width: 360px; height: 360px; bottom: -14%; left: 20%;
    background: radial-gradient(circle, {EMOTION_META['joy']['c1']}, transparent 70%);
    animation: floatC 22s ease-in-out infinite; }}

@keyframes floatA {{
    0%, 100% {{ transform: translate(0, 0) scale(1); }}
    50% {{ transform: translate(50px, 40px) scale(1.12); }}
}}
@keyframes floatB {{
    0%, 100% {{ transform: translate(0, 0) scale(1); }}
    50% {{ transform: translate(-40px, 30px) scale(1.08); }}
}}
@keyframes floatC {{
    0%, 100% {{ transform: translate(0, 0) scale(1); }}
    50% {{ transform: translate(30px, -35px) scale(1.15); }}
}}
@media (prefers-reduced-motion: reduce) {{
    .blob {{ animation: none !important; }}
}}

section[data-testid="stSidebar"] {{
    background: rgba(12, 10, 22, 0.6);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    border-right: 1px solid {COLORS['border']};
}}

h1, h2, h3, .display-font {{
    font-family: 'Space Grotesk', sans-serif;
}}

.mono {{
    font-family: 'JetBrains Mono', monospace;
}}

/* ---- Hero ---- */
.hero-eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    font-size: 0.72rem;
    color: {COLORS['muted']};
    margin-bottom: 0.4rem;
}}
.hero-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1.1;
    margin: 0 0 0.6rem 0;
    background: linear-gradient(90deg, #8FB4FF, #FF9FC7, #FFE066, #8FB4FF);
    background-size: 300% auto;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: gradientShift 10s ease infinite;
}}
@keyframes gradientShift {{
    0% {{ background-position: 0% center; }}
    50% {{ background-position: 100% center; }}
    100% {{ background-position: 0% center; }}
}}
.hero-sub {{
    color: {COLORS['muted']};
    font-size: 1.02rem;
    max-width: 640px;
    line-height: 1.6;
}}

/* ---- Glass panels ---- */
.panel {{
    background: {COLORS['glass']};
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    border: 1px solid {COLORS['border']};
    border-radius: 22px;
    padding: 1.8rem 1.9rem;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
}}

textarea {{
    font-family: 'Inter', sans-serif !important;
    background: rgba(255,255,255,0.05) !important;
    color: {COLORS['text']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 14px !important;
}}
textarea:focus {{
    border-color: rgba(255,255,255,0.35) !important;
    box-shadow: 0 0 0 3px rgba(124,92,255,0.18) !important;
}}

/* ---- Buttons ---- */
button[kind="primary"] {{
    background: linear-gradient(135deg, #7C5CFF, #FF6FA5 60%, #FFA928) !important;
    background-size: 180% 180% !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 999px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    font-size: 0.88rem !important;
    padding: 0.7rem 1.7rem !important;
    box-shadow: 0 8px 28px rgba(124,92,255,0.4);
    transition: transform 0.18s ease, box-shadow 0.18s ease, background-position 0.4s ease;
}}
button[kind="primary"]:hover {{
    transform: translateY(-2px);
    background-position: 100% 50% !important;
    box-shadow: 0 12px 36px rgba(255,111,165,0.45);
    color: #ffffff !important;
}}

button[kind="secondary"] {{
    background: {COLORS['glass_alt']} !important;
    color: {COLORS['muted']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 999px !important;
    font-family: 'Inter', sans-serif !important;
    text-transform: none !important;
    font-weight: 400 !important;
    font-size: 0.82rem !important;
    padding: 0.4rem 1rem !important;
    backdrop-filter: blur(10px);
    transition: border-color 0.15s ease, color 0.15s ease;
}}
button[kind="secondary"]:hover {{
    border-color: rgba(255,255,255,0.4) !important;
    color: {COLORS['text']} !important;
}}

/* ---- Result hero: the label number ---- */
.result-wrap {{
    display: flex;
    align-items: center;
    gap: 1.5rem;
    margin-bottom: 1.7rem;
}}
.label-badge {{
    width: 112px;
    height: 112px;
    min-width: 112px;
    border-radius: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: #ffffff;
    text-shadow: 0 2px 12px rgba(0,0,0,0.25);
}}
.result-emotion {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 600;
    text-transform: capitalize;
    margin: 0;
}}
.result-caption {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {COLORS['muted']};
    margin: 0.2rem 0 0 0;
}}

/* ---- Spectrum meter ---- */
.meter-row {{
    display: grid;
    grid-template-columns: 100px 1fr 56px;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.6rem;
}}
.meter-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: {COLORS['muted']};
    text-transform: capitalize;
}}
.meter-track {{
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    height: 11px;
    overflow: hidden;
    border: 1px solid {COLORS['border']};
}}
.meter-fill {{
    height: 100%;
    border-radius: 8px;
    transition: width 0.7s cubic-bezier(0.22, 1, 0.36, 1);
}}
.meter-pct {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: {COLORS['muted']};
    text-align: right;
}}

.footer-note {{
    color: {COLORS['muted']};
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 2rem;
    border-top: 1px solid {COLORS['border']};
    padding-top: 1rem;
}}

hr {{ border-color: {COLORS['border']}; }}
</style>

<div class="aurora-bg">
    <div class="blob blob-sadness"></div>
    <div class="blob blob-anger"></div>
    <div class="blob blob-love"></div>
    <div class="blob blob-surprise"></div>
    <div class="blob blob-fear"></div>
    <div class="blob blob-joy"></div>
</div>
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
    st.markdown('<p class="display-font" style="font-size:1.25rem; font-weight:600; margin-top:-0.3rem;">TF-IDF + Logistic Regression</p>', unsafe_allow_html=True)

    st.metric("Held-out accuracy", f"{artifacts['accuracy']*100:.1f}%")
    st.metric("Training sentences", f"{artifacts['n_rows']:,}")

    st.markdown("---")
    st.markdown('<p class="hero-eyebrow">Label Key</p>', unsafe_allow_html=True)
    for i in range(6):
        emo = artifacts["id_to_emotion"][i]
        st.markdown(
            f"""<div style="display:flex; align-items:center; gap:0.55rem; margin-bottom:0.5rem;">
                <div style="width:12px; height:12px; border-radius:4px; background:{grad(emo, 135)};"></div>
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
        pct = count / max_count * 100
        st.markdown(
            f"""<div style="margin-bottom:0.55rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:{COLORS['muted']}; margin-bottom:3px;">
                    <span style="text-transform:capitalize;">{emo}</span><span class="mono">{count}</span>
                </div>
                <div style="background:rgba(255,255,255,0.06); border-radius:6px; height:7px; overflow:hidden;">
                    <div style="background:{grad(emo, 90)}; width:{pct}%; height:100%;"></div>
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
        emo = result["emotion"]
        meta = EMOTION_META[emo]

        st.markdown('<p class="hero-eyebrow">Result</p>', unsafe_allow_html=True)
        st.markdown(
            f"""<div class="result-wrap">
                <div class="label-badge" style="background:{grad(emo)}; box-shadow: 0 12px 32px {meta['c1']}55;">{result['label']}</div>
                <div>
                    <p class="result-emotion">{meta['glyph']} {emo}</p>
                    <p class="result-caption">predicted label · {result['proba'][result['label']]*100:.1f}% confidence</p>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        order = np.argsort(result["proba"])[::-1]
        for idx in order:
            row_emo = artifacts["id_to_emotion"][idx]
            pct = result["proba"][idx] * 100
            st.markdown(
                f"""<div class="meter-row">
                    <span class="meter-label">{row_emo}</span>
                    <div class="meter-track">
                        <div class="meter-fill" style="width:{pct:.1f}%; background:{grad(row_emo, 90)};"></div>
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
