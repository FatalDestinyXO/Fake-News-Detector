"""
train.py
--------
Train and compare three classifiers for the Fake News Detection task:
    1. Logistic Regression
    2. Multinomial Naive Bayes
    3. Random Forest

Each model is evaluated on the same hold-out test set using:
    - accuracy, precision, recall, F1-score
    - confusion-matrix plot (saved to /models)

The TF-IDF vectorizer and every trained model are pickled into the
`models/` folder so the Streamlit app can re-use them at inference time.

Run:
    python train.py
"""

import json
import os
import pathlib

import joblib
import matplotlib

matplotlib.use("Agg")  # headless plotting (no display required)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocess import clean_series

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRUE_PATH = DATA_DIR / "True.csv"
FAKE_PATH = DATA_DIR / "Fake.csv"
SINGLE_PATH = DATA_DIR / "fake_news.csv"  # fallback for the old single-CSV format
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load the Kaggle Fake-News dataset.

    Two layouts are supported:

    A) Two-file layout (recommended) – the official Kaggle release:
       - data/True.csv  → real news  → label = 0
       - data/Fake.csv  → fake news  → label = 1
       Columns: title, text, subject, date

    B) Single-file fallback – data/fake_news.csv with columns
       id, title, author, text, label.

    In both cases we build a unified 'content' column = title + " " + text,
    which is the input feature used by the TF-IDF vectorizer.
    """
    if TRUE_PATH.exists() and FAKE_PATH.exists():
        print(f"[1/5] Loading two-file Kaggle layout:")
        print(f"      • {TRUE_PATH.name}  (label=0, real)")
        print(f"      • {FAKE_PATH.name}  (label=1, fake)")

        true_df = pd.read_csv(TRUE_PATH)
        fake_df = pd.read_csv(FAKE_PATH)
        true_df["label"] = 0
        fake_df["label"] = 1

        df = pd.concat([true_df, fake_df], ignore_index=True)
        # Shuffle so the two classes are interleaved (important for any
        # downstream code that doesn't shuffle itself).
        df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    elif SINGLE_PATH.exists():
        print(f"[1/5] Loading single-file layout: {SINGLE_PATH}")
        df = pd.read_csv(SINGLE_PATH)
        if "label" not in df.columns:
            raise ValueError("fake_news.csv must contain a 'label' column.")

    else:
        raise FileNotFoundError(
            "No dataset found. Place either True.csv + Fake.csv "
            f"OR fake_news.csv inside {DATA_DIR}/."
        )

    # Build the input feature: title + text (drop NaN).
    title = df.get("title", pd.Series([""] * len(df))).fillna("")
    text = df.get("text", pd.Series([""] * len(df))).fillna("")
    df["content"] = (title.astype(str) + " " + text.astype(str)).str.strip()

    # Drop empty rows after concatenation.
    df = df[df["content"].str.len() > 0].reset_index(drop=True)

    print(f"      → total rows: {len(df):,}  | real (0): "
          f"{(df['label']==0).sum():,}  | fake (1): {(df['label']==1).sum():,}")
    return df


# ---------------------------------------------------------------------------
# 2. Train / Eval helpers
# ---------------------------------------------------------------------------
def evaluate(name: str, model, X_test, y_test) -> dict:
    """Return a dict of metrics + save a confusion-matrix plot."""
    preds = model.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }

    # Pretty per-class report in the console for transparency.
    print(f"\n--- {name} ---")
    print(classification_report(y_test, preds, target_names=["Real", "Fake"]))

    # Confusion-matrix plot (saved as PNG).
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name} – Confusion Matrix")
    fig.tight_layout()
    out = MODELS_DIR / f"cm_{name.lower().replace(' ', '_')}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"      → confusion matrix saved to {out.name}")

    return metrics


def main() -> None:
    # -----------------------------------------------------------------------
    # 1. Load and split
    # -----------------------------------------------------------------------
    df = load_data()

    print("[2/5] Cleaning text (lowercase, stop-words, punctuation)…")
    df["clean"] = clean_series(df["content"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"],
        df["label"].astype(int),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    # -----------------------------------------------------------------------
    # 2. TF-IDF vectorize
    # -----------------------------------------------------------------------
    print("[3/5] Fitting TF-IDF vectorizer (ngram=(1,2), max_features=10000)…")
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # -----------------------------------------------------------------------
    # 3. Define candidate models
    # -----------------------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Naive Bayes": MultinomialNB(alpha=0.3),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
        ),
    }

    # -----------------------------------------------------------------------
    # 4. Train + evaluate
    # -----------------------------------------------------------------------
    print("[4/5] Training models…")
    results = []
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        metrics = evaluate(name, model, X_test_vec, y_test)
        results.append(metrics)

        slug = name.lower().replace(" ", "_")
        joblib.dump(model, MODELS_DIR / f"{slug}.pkl")
        print(f"      → saved {slug}.pkl")

    # -----------------------------------------------------------------------
    # 5. Persist vectorizer + background sample + summary
    # -----------------------------------------------------------------------
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")

    # Small background sample used by SHAP's TreeExplainer (interventional mode)
    # for token-level explanations in the Streamlit app.
    rng = np.random.default_rng(RANDOM_STATE)
    bg_size = min(100, X_train_vec.shape[0])
    bg_idx = rng.choice(X_train_vec.shape[0], size=bg_size, replace=False)
    bg = X_train_vec[bg_idx].toarray().astype(np.float32)
    np.save(MODELS_DIR / "shap_background.npy", bg)
    print(f"      → SHAP background sample saved ({bg.shape})")

    results_df = pd.DataFrame(results).set_index("model").round(4)
    print("\n[5/5] Final comparison:")
    print(results_df.to_string())

    best = results_df["f1"].idxmax()
    print(f"\n✅ Best model by F1-score: {best} "
          f"(F1={results_df.loc[best, 'f1']:.4f})")

    # Save a JSON summary for the Streamlit app to display.
    summary = {
        "metrics": results_df.reset_index().to_dict(orient="records"),
        "best_model": best,
        "n_train": int(X_train_vec.shape[0]),
        "n_test": int(X_test_vec.shape[0]),
        "n_features": int(X_train_vec.shape[1]),
    }
    with open(MODELS_DIR / "results.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"      → summary saved to models/results.json")


if __name__ == "__main__":
    main()
