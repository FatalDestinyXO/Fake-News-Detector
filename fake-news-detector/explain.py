"""
explain.py
----------
Token-level SHAP explanations for the three trained models.

Approach:
    • Logistic Regression — linear model. SHAP values under the independent-
      feature assumption are exactly  ``coef[i] * x[i]``  (the same closed-form
      result that ``shap.LinearExplainer`` returns with a zero baseline).
    • Multinomial Naive Bayes — the log-odds  log P(fake|x) / P(real|x) is also
      linear in the TF-IDF features, with coefficients
      ``feature_log_prob_[fake] - feature_log_prob_[real]``. So per-feature
      contributions are  ``(log_prob_fake[i] - log_prob_real[i]) * x[i]``.
      Mathematically identical to a SHAP linear explainer for this model.
    • Random Forest — non-linear, so we use ``shap.TreeExplainer`` which
      computes exact SHAP values for tree ensembles.

All three return the same shape: a list of ``(token, contribution)`` tuples
where positive contributions push the prediction towards **Fake** (class 1)
and negative contributions push it towards **Real** (class 0).
"""

from __future__ import annotations

import pathlib

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

_MODELS_DIR = pathlib.Path(__file__).resolve().parent / "models"
_BACKGROUND_CACHE: np.ndarray | None = None


def _load_background() -> np.ndarray:
    """Return (and lazily cache) the SHAP background sample saved by train.py."""
    global _BACKGROUND_CACHE
    if _BACKGROUND_CACHE is None:
        path = _MODELS_DIR / "shap_background.npy"
        if not path.exists():
            raise FileNotFoundError(
                "SHAP background sample is missing. Re-run `python train.py` "
                "to regenerate models/shap_background.npy."
            )
        _BACKGROUND_CACHE = np.load(path)
    return _BACKGROUND_CACHE


# A single global cache for the (slow-to-build) TreeExplainer per RF instance.
_TREE_EXPLAINERS: dict[int, shap.TreeExplainer] = {}


def _per_feature_contributions(model, x_sparse) -> np.ndarray:
    """Return a 1-D numpy array of length n_features with the contribution of
    each feature to the **Fake** (class 1) log-odds for this single input."""
    if isinstance(model, LogisticRegression):
        # coef_ has shape (1, n_features) for binary classification.
        return x_sparse.multiply(model.coef_[0]).toarray().ravel()

    if isinstance(model, MultinomialNB):
        # feature_log_prob_ shape: (n_classes, n_features); classes_ tells order.
        classes = list(model.classes_)
        fake_idx = classes.index(1)
        real_idx = classes.index(0)
        log_odds = model.feature_log_prob_[fake_idx] - model.feature_log_prob_[real_idx]
        return x_sparse.multiply(log_odds).toarray().ravel()

    if isinstance(model, RandomForestClassifier):
        key = id(model)
        explainer = _TREE_EXPLAINERS.get(key)
        if explainer is None:
            # Interventional mode + a small background sample produces
            # well-scaled SHAP values (in probability space). Without a
            # background, SHAP can return numerically unstable outputs on
            # high-dim sparse-derived inputs.
            bg = _load_background()
            explainer = shap.TreeExplainer(
                model,
                data=bg,
                feature_perturbation="interventional",
                model_output="probability",
            )
            _TREE_EXPLAINERS[key] = explainer

        sv = explainer.shap_values(x_sparse.toarray(), check_additivity=False)
        # SHAP versions differ in output shape:
        #   • list of arrays per class  → sv[1] for the Fake class
        #   • single ndarray (n, f, c)  → sv[0, :, 1]
        #   • single ndarray (n, f)     → sv[0]  (already class-1 contributions)
        if isinstance(sv, list):
            arr = sv[1][0]
        else:
            arr = np.asarray(sv)
            if arr.ndim == 3:
                arr = arr[0, :, 1]
            elif arr.ndim == 2:
                arr = arr[0]
        return np.asarray(arr).ravel()

    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def top_token_contributions(
    model,
    vectorizer,
    cleaned_text: str,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Return the top-k tokens (by absolute contribution) for this input.

    Parameters
    ----------
    model        : a fitted sklearn classifier (LR, NB or RF).
    vectorizer   : the fitted TF-IDF vectorizer used at training time.
    cleaned_text : already pre-processed text (see ``preprocess.clean_text``).
    top_k        : number of tokens to keep.

    Returns
    -------
    List of ``(token, contribution)``:
        positive contribution → pushes prediction toward **Fake**
        negative contribution → pushes prediction toward **Real**
    """
    x = vectorizer.transform([cleaned_text])
    if x.nnz == 0:
        return []

    contrib = _per_feature_contributions(model, x)
    feature_names = vectorizer.get_feature_names_out()

    # Restrict to features that are actually present in the input.
    nonzero_idx = x.nonzero()[1]
    tokens = [(feature_names[i], float(contrib[i])) for i in nonzero_idx]

    tokens.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return tokens[:top_k]
