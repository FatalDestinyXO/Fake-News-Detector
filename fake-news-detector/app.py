"""
app.py
------
Streamlit front-end for the Fake News Detector.

Run locally with:
    streamlit run app.py
"""

import json
import pathlib

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from explain import top_token_contributions
from preprocess import clean_text

# ---------------------------------------------------------------------------
# Page config – must be the very first Streamlit command.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


# ---------------------------------------------------------------------------
# Loaders – cached so the heavy .pkl files only deserialize once per session.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained models…")
def load_artifacts():
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    models = {
        "Logistic Regression": joblib.load(MODELS_DIR / "logistic_regression.pkl"),
        "Naive Bayes": joblib.load(MODELS_DIR / "naive_bayes.pkl"),
        "Random Forest": joblib.load(MODELS_DIR / "random_forest.pkl"),
    }
    with open(MODELS_DIR / "results.json") as fh:
        summary = json.load(fh)
    return vectorizer, models, summary


def predict(model, vectorizer, text: str):
    """Return (label, confidence, proba, cleaned) for a single text input."""
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    # All three sklearn models implement predict_proba.
    proba = model.predict_proba(vec)[0]
    label_idx = int(proba.argmax())
    confidence = float(proba[label_idx])
    label = "Fake" if label_idx == 1 else "Real"
    return label, confidence, proba, cleaned


def render_token_chart(contributions: list[tuple[str, float]]):
    """Render a horizontal bar chart of token contributions.

    Red bars (positive)  → tokens pushing the prediction toward FAKE.
    Green bars (negative) → tokens pushing the prediction toward REAL.
    """
    if not contributions:
        st.info("No vocabulary tokens from the input matched the model's "
                "feature space – cannot explain.")
        return

    # Sort so the largest |contribution| sits at the top of the chart.
    contributions = sorted(contributions, key=lambda kv: kv[1])
    tokens = [t for t, _ in contributions]
    values = [v for _, v in contributions]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig, ax = plt.subplots(figsize=(6, max(3, 0.35 * len(tokens))))
    ax.barh(tokens, values, color=colors, edgecolor="white")
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("Contribution  (←  Real        Fake  →)")
    ax.set_title("Top tokens driving this prediction")
    ax.tick_params(axis="y", labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📰 Fake News Detector")
st.caption(
    "Paste a news headline or article below and let three Machine-Learning "
    "models decide whether it looks **Real** or **Fake**."
)

# ---- Sidebar ----------------------------------------------------------------
if not (MODELS_DIR / "tfidf_vectorizer.pkl").exists():
    st.error(
        "❌ Models not found. Please run `python train.py` first to train the "
        "models and create the `models/` directory."
    )
    st.stop()

vectorizer, models, summary = load_artifacts()

with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Choose a model",
        list(models.keys()),
        index=list(models.keys()).index(summary["best_model"]),
        help=f"Default = best performer ({summary['best_model']}).",
    )

    st.divider()
    st.subheader("📊 Model performance")
    metrics_df = pd.DataFrame(summary["metrics"]).set_index("model").round(4)
    st.dataframe(metrics_df, use_container_width=True)
    st.success(f"🏆 Best F1: **{summary['best_model']}**")

    st.caption(
        f"Train: {summary['n_train']:,} · Test: {summary['n_test']:,} · "
        f"Features: {summary['n_features']:,}"
    )

# ---- Main area --------------------------------------------------------------
col_input, col_result = st.columns([3, 2])

with col_input:
    st.subheader("📝 Your news text")
    default = (
        "Breaking: Scientists discover that drinking coffee while standing on "
        "one foot cures every disease within five minutes, according to an "
        "unnamed source."
    )
    user_text = st.text_area(
        "Headline or article",
        value=default,
        height=260,
        help="Paste full article text for better accuracy.",
    )
    submitted = st.button("🔍 Analyse", type="primary", use_container_width=True)

with col_result:
    st.subheader("🎯 Prediction")
    if submitted:
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            model = models[model_name]
            label, confidence, proba, cleaned = predict(model, vectorizer, user_text)

            if label == "Fake":
                st.error(f"🚨 Predicted: **FAKE** ({confidence:.1%} confidence)")
            else:
                st.success(f"✅ Predicted: **REAL** ({confidence:.1%} confidence)")

            st.progress(confidence)
            st.write("**Class probabilities**")
            st.bar_chart(
                pd.DataFrame(
                    {"probability": proba},
                    index=["Real (0)", "Fake (1)"],
                )
            )
            st.caption(f"Model used: `{model_name}`")

            # Cache the explanation context so we can render it below the row.
            st.session_state["explain_ctx"] = {
                "model_name": model_name,
                "cleaned": cleaned,
            }
    else:
        st.info("Click **Analyse** to get a prediction.")

# ---- SHAP token-level explanation ------------------------------------------
ctx = st.session_state.get("explain_ctx")
if ctx is not None:
    st.divider()
    st.subheader("🔍 Why did the model decide this?")
    st.caption(
        "Top 10 input tokens ranked by their SHAP contribution. "
        "🔴 red = pushes toward **Fake**,  🟢 green = pushes toward **Real**."
    )
    with st.spinner("Computing SHAP explanations…"):
        contribs = top_token_contributions(
            models[ctx["model_name"]],
            vectorizer,
            ctx["cleaned"],
            top_k=10,
        )
    render_token_chart(contribs)

# ---- Confusion-matrix gallery ----------------------------------------------
st.divider()
st.subheader("🧮 Confusion matrices on the hold-out test set")
cm_cols = st.columns(3)
for col, (name, _) in zip(cm_cols, models.items()):
    slug = name.lower().replace(" ", "_")
    img_path = MODELS_DIR / f"cm_{slug}.png"
    with col:
        st.markdown(f"**{name}**")
        if img_path.exists():
            st.image(str(img_path), use_column_width=True)
        else:
            st.caption("No confusion matrix found – re-run `train.py`.")

st.divider()
with st.expander("ℹ️ About this project"):
    st.markdown(
        """
        **Pipeline**
        1. Text cleaning (lower-case, strip URLs/HTML/punctuation, stop-word removal).
        2. TF-IDF vectorization (uni- + bi-grams, max 10k features).
        3. Train **Logistic Regression**, **Naive Bayes**, **Random Forest**.
        4. Compare with accuracy, precision, recall, F1 and confusion matrices.
        5. Best-performing model is selected as the default for the UI.

        Source code: `train.py`, `preprocess.py`, `app.py`.
        """
    )
