"""Multilingual sentiment (works for Cantonese/Chinese, Korean, English)."""
import numpy as np
from transformers import pipeline

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
_clf = None


def _get():
    global _clf
    if _clf is None:
        _clf = pipeline("sentiment-analysis", model=MODEL, truncation=True, max_length=256)
    return _clf


def score_texts(texts, batch_size=32):
    """Returns a score in [-1, 1] per text: P(positive) - P(negative)."""
    if len(texts) == 0:
        return np.array([])
    clf = _get()
    out = clf([t or "" for t in texts], batch_size=batch_size, top_k=None)
    scores = []
    for probs in out:
        p = {d["label"].lower(): d["score"] for d in probs}
        scores.append(p.get("positive", 0) - p.get("negative", 0))
    return np.array(scores)


def engagement_weight(df):
    """Posts with more reach count more (log-scaled)."""
    reach = df[["likes", "reposts", "views", "followers"]].fillna(0)
    return 1 + np.log1p(reach["likes"] + 2 * reach["reposts"] + 0.01 * reach["views"] + 0.001 * reach["followers"])
