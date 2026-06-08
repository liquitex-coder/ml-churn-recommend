"""Support-text analysis (FR-3, FR-4).

Embeds support-inquiry text and measures cosine similarity to a set of
"解約匂わせ / dissatisfaction" reference cases, yielding a per-customer
``churn_hint_score`` that is later folded into the churn model (FR-4).

Default embedding is TF-IDF (sklearn, fully offline). A Sentence-Transformers
path is optional and guarded behind a try/except import so the demo still runs
when that package is unavailable (NFR-2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config

# Reference "解約匂わせ" / dissatisfaction cases used as the similarity anchor (FR-3).
REFERENCE_CASES = [
    "解約したい 解約方法を教えて",
    "他社に乗り換えを検討しています",
    "料金が高いので解約を考えています",
    "サービスに不満があり継続するか迷っています",
    "I want to cancel my subscription, too expensive",
]


def _char_ngram_vectorizer() -> TfidfVectorizer:
    """TF-IDF over character n-grams.

    Character n-grams work well for mixed JP/EN text without a tokenizer,
    keeping the demo dependency-free (NFR-2).
    """
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)


def embed_texts_tfidf(texts: list[str], reference_cases: list[str] | None = None):
    """Embed ``texts`` and ``reference_cases`` in a shared TF-IDF space (FR-3)."""
    reference_cases = reference_cases or REFERENCE_CASES
    vec = _char_ngram_vectorizer()
    matrix = vec.fit_transform(list(texts) + list(reference_cases))
    n_texts = len(texts)
    return matrix[:n_texts], matrix[n_texts:], vec


def embed_texts_sentence_transformers(texts: list[str], reference_cases: list[str] | None = None):
    """Optional Sentence-Transformers embedding path (FR-3).

    Imported lazily and guarded so the default TF-IDF path is unaffected when
    the package is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "sentence-transformers is not installed; use the default TF-IDF path."
        ) from exc

    reference_cases = reference_cases or REFERENCE_CASES
    model = SentenceTransformer("all-MiniLM-L6-v2")
    text_emb = model.encode(list(texts), normalize_embeddings=True)
    ref_emb = model.encode(list(reference_cases), normalize_embeddings=True)
    return text_emb, ref_emb, model


def text_similarity_scores(
    texts: list[str],
    reference_cases: list[str] | None = None,
    use_sentence_transformers: bool = False,
) -> np.ndarray:
    """Per-text max cosine similarity to the reference cancellation cases (FR-3).

    Returns an array in [0, 1], one value per input text.
    """
    if not texts:
        return np.zeros(0, dtype=float)

    if use_sentence_transformers:
        text_emb, ref_emb, _ = embed_texts_sentence_transformers(texts, reference_cases)
        sims = cosine_similarity(np.asarray(text_emb), np.asarray(ref_emb))
    else:
        text_mat, ref_mat, _ = embed_texts_tfidf(texts, reference_cases)
        sims = cosine_similarity(text_mat, ref_mat)

    # Strongest match to any reference case; clip negatives from ST cosine.
    return np.clip(sims.max(axis=1), 0.0, 1.0)


def compute_churn_hint_scores(
    support_texts: pd.DataFrame,
    customers: pd.DataFrame,
    use_sentence_transformers: bool = False,
) -> pd.DataFrame:
    """Aggregate text similarity to a per-customer ``churn_hint_score`` (FR-3, FR-4).

    Returns a frame with columns ``[customer_id, churn_hint_score]`` covering
    *every* customer (0.0 when a customer has no support texts).
    """
    texts = support_texts["text"].astype(str).tolist()
    sims = text_similarity_scores(texts, use_sentence_transformers=use_sentence_transformers)

    scored = support_texts[["customer_id"]].copy()
    scored["sim"] = sims
    # A customer's hint score = strongest cancellation-like signal across texts.
    per_customer = scored.groupby("customer_id")["sim"].max().reset_index()
    per_customer = per_customer.rename(columns={"sim": config.TEXT_FEATURE})

    out = customers[["customer_id"]].merge(per_customer, on="customer_id", how="left")
    out[config.TEXT_FEATURE] = out[config.TEXT_FEATURE].fillna(0.0)
    return out
