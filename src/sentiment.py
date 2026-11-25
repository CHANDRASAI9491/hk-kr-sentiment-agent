"""Multilingual sentiment (works for Cantonese/Chinese, Korean, English)."""
import numpy as np
import pandas as pd
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
    if df is None or len(df) == 0:
        return np.array([])
    likes = pd.to_numeric(df.get("likes", 0), errors="coerce").fillna(0).clip(lower=0)
    reposts = pd.to_numeric(df.get("reposts", 0), errors="coerce").fillna(0).clip(lower=0)
    views = pd.to_numeric(df.get("views", 0), errors="coerce").fillna(0).clip(lower=0)
    followers = pd.to_numeric(df.get("followers", 0), errors="coerce").fillna(0).clip(lower=0)
    score = likes + 2 * reposts + 0.01 * views + 0.001 * followers
    return 1.0 + np.log1p(score.to_numpy(dtype=float))
