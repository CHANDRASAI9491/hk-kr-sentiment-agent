"""Aggregate posts and news to daily features aligned with price bars."""
import numpy as np
import pandas as pd

from .sentiment import engagement_weight


def daily_sentiment(posts, news):
    frames = []
    if posts is not None and not posts.empty:
        p = posts.copy()
        p["w"] = engagement_weight(p)
        frames.append(p[["ticker", "source", "timestamp", "sentiment", "w"]])
    if news is not None and not news.empty:
        n = news.copy()
        n["w"] = 1.5  # news weighted a bit above an average post
        frames.append(n[["ticker", "source", "timestamp", "sentiment", "w"]])
    if not frames:
        return pd.DataFrame()
    allp = pd.concat(frames)
    allp["date"] = pd.to_datetime(allp["timestamp"], utc=True).dt.normalize()
    g = allp.groupby(["ticker", "date"])
    def _agg(d):
        w = d["w"].values
        w_sum = np.sum(w)
        sent_w = np.average(d["sentiment"], weights=w) if w_sum > 0 else d["sentiment"].mean()
        return pd.Series({
            "sent_w": float(sent_w),
            "sent_mean": float(d["sentiment"].mean()),
            "mentions": len(d),
            "bull_ratio": float((d["sentiment"] > 0.2).mean()),
        })
    out = g.apply(_agg, include_groups=False).reset_index()
    return out


def build_dataset(prices, sent):
    """Join price features with that day's sentiment; target = next close return."""
    df = prices.copy()
    df["ret_1"] = df["close"].pct_change()
    df["ret_5"] = df["close"].pct_change(5)
    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["target_ret"] = df["close"].shift(-1) / df["close"] - 1
    df = df.join(sent.set_index("date"), how="left") if not sent.empty else df
    for c in ["sent_w", "sent_mean", "mentions", "bull_ratio"]:
        if c not in df:
            df[c] = 0.0
        df[c] = df[c].fillna(0)
    df["sent_chg"] = df["sent_w"].diff().fillna(0)
    return df
