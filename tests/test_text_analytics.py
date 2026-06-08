# Requirements traceability for this file:
#   FR-3 vectorize support text + similarity to "解約匂わせ"/dissatisfaction cases
#   FR-4 fold text-similarity into a per-customer churn-hint correction feature
#
# covers: FR-3
# covers: FR-4
from __future__ import annotations

import numpy as np

from churn_recommend import config, data_gen, text_analytics


def test_similarity_higher_for_cancellation_text():
    # covers: FR-3  -- cancellation-hint text scores higher than satisfied text
    texts = [
        "解約したい 解約方法を教えてください",  # cancellation hint
        "とても満足しています ありがとうございます",  # positive
    ]
    sims = text_analytics.text_similarity_scores(texts)
    assert sims.shape == (2,)
    assert sims[0] > sims[1]
    assert np.all((sims >= 0.0) & (sims <= 1.0))


def test_churn_hint_score_per_customer_for_feature():
    # covers: FR-4  -- produces a churn_hint_score covering every customer
    tables = data_gen.generate_all(n_customers=80, n_texts=120, seed=7)
    scores = text_analytics.compute_churn_hint_scores(
        tables["support_texts"], tables["customers"]
    )
    assert config.TEXT_FEATURE in scores.columns
    # one row per customer, all scores in [0, 1]
    assert len(scores) == len(tables["customers"])
    vals = scores[config.TEXT_FEATURE].to_numpy()
    assert np.all((vals >= 0.0) & (vals <= 1.0))


def test_optional_sentence_transformers_guarded():
    # covers: FR-3  -- optional ST path is guarded; default TF-IDF still works
    # sentence-transformers is not installed in this env, so the guard raises.
    try:
        text_analytics.embed_texts_sentence_transformers(["解約したい"])
        raised = False
    except ImportError:
        raised = True
    assert raised
    # default path remains usable
    assert text_analytics.text_similarity_scores(["解約したい"]).shape == (1,)
