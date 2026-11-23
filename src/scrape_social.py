"""Social post ingestion through Apify actors (one actor per source)."""
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from apify_client import ApifyClient

# Different actors name their fields differently; map the common ones here.
FIELD_MAP = {
    "timestamp": ["timestamp", "date", "createdAt", "created_at", "postedAt", "time"],
    "text": ["text", "content", "message", "body", "title"],
    "likes": ["likes", "likeCount", "like_count", "upvotes", "reactions"],
    "reposts": ["reposts", "shares", "forwards", "retweetCount"],
    "views": ["views", "viewCount", "view_count"],
    "followers": ["followers", "authorFollowers", "subscribers", "author_followers"],
    "author": ["author", "username", "user", "authorName"],
}


def _pick(item, keys, default=None):
    for k in keys:
        if k in item and item[k] not in (None, ""):
            return item[k]
    return default


def _fill(obj, ticker):
    """Replace {query}/{code} placeholders in the actor input."""
    code = ticker["symbol"].split(".")[0]
    query = " OR ".join(ticker["keywords"])
    if isinstance(obj, dict):
        return {k: _fill(v, ticker) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fill(v, ticker) for v in obj]
    if isinstance(obj, str):
        return obj.replace("{query}", query).replace("{code}", code)
    return obj


def scrape(exchange, ticker, source, lookback_days):
    client = ApifyClient(os.environ["APIFY_TOKEN"])
    run = client.actor(source["actor_id"]).call(run_input=_fill(source["run_input"], ticker))
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())

    rows = []
    for it in items:
        rows.append({
            "exchange": exchange,
            "ticker": ticker["symbol"],
            "source": source["name"],
            "timestamp": _pick(it, FIELD_MAP["timestamp"]),
            "text": _pick(it, FIELD_MAP["text"], ""),
            "likes": _pick(it, FIELD_MAP["likes"], 0),
            "reposts": _pick(it, FIELD_MAP["reposts"], 0),
            "views": _pick(it, FIELD_MAP["views"], 0),
            "followers": _pick(it, FIELD_MAP["followers"], 0),
            "author": _pick(it, FIELD_MAP["author"], ""),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Align everything to UTC and keep only the lookback window.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    df = df[df["timestamp"] >= cutoff].dropna(subset=["timestamp"])
    for c in ["likes", "reposts", "views", "followers"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df
