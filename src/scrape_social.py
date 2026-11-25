"""Social post ingestion through Apify actors.

Sources (2 per exchange):
  HKEX: LIHKG forum (keyword search)      -> claude_code_reviewer/lihkg-scraper-en
        Telegram finance channels          -> automation-lab/telegram-scraper
  KRX:  Naver Finance 종목토론실 (by code)  -> hgservices/naver-stock-discussion-scraper
        Telegram finance channels          -> automation-lab/telegram-scraper

Every adapter returns the same columns:
exchange, ticker, source, timestamp (UTC), text, likes, reposts, views, followers, author, url
"""
import os
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
from apify_client import ApifyClient

COLS = ["exchange", "ticker", "source", "timestamp", "text", "likes",
        "reposts", "views", "followers", "author", "url"]

_client = None
_channel_cache = {}  # Telegram channels are scraped once per exchange, then filtered per ticker


def client():
    global _client
    if _client is None:
        token = os.environ.get("APIFY_TOKEN", "")
        if token and not token.startswith("apify_api_"):
            token = f"apify_api_{token}"
        _client = ApifyClient(token)
    return _client


def run_actor(actor_id, run_input):
    run = client().actor(actor_id).call(run_input=run_input)
    ds = run["defaultDatasetId"] if isinstance(run, dict) else run.default_dataset_id
    return list(client().dataset(ds).iterate_items())


def to_utc(value):
    """Parse ISO timestamps, and also 'Wed 2026-11-25 4:15 PM IST (UTC+05:30)' style strings."""
    if value in (None, ""):
        return pd.NaT
    s = str(value)
    m = re.search(r"(\d{4}-\d{2}-\d{2}) (\d{1,2}:\d{2} [AP]M) .*UTC([+-]\d{2}:\d{2})", s)
    if m:
        dt = datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %I:%M %p")
        sign = 1 if m.group(3)[0] == "+" else -1
        hh, mm = map(int, m.group(3)[1:].split(":"))
        return pd.Timestamp(dt - sign * timedelta(hours=hh, minutes=mm), tz="UTC")
    return pd.to_datetime(s, utc=True, errors="coerce")


def _frame(rows, lookback_days):
    df = pd.DataFrame(rows, columns=COLS)
    if df.empty:
        return df
    df["timestamp"] = df["timestamp"].map(to_utc)
    cutoff = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=lookback_days))
    df = df.dropna(subset=["timestamp"])
    df = df[df["timestamp"] >= cutoff]
    for c in ["likes", "reposts", "views", "followers"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["text"] = df["text"].fillna("").astype(str).str.slice(0, 1000)
    return df[df["text"].str.strip() != ""].drop_duplicates(subset=["text"])


# ---------- adapters ----------

def lihkg(exchange, ticker, src, lookback_days):
    items = run_actor(src["actor_id"], {
        "mode": "search",
        "keywords": ticker["keywords"][:2],
        "dateFrom": f"{lookback_days}d",
        "maxItems": src.get("max_items", 40),
    })
    rows = [{
        "exchange": exchange, "ticker": ticker["symbol"], "source": "lihkg",
        "timestamp": it.get("created_at"),
        "text": " ".join(x for x in [it.get("title"), it.get("content")] if x),
        "likes": (it.get("like_count") or 0) - (it.get("dislike_count") or 0),
        "reposts": it.get("reply_count") or 0, "views": 0, "followers": 0,
        "author": it.get("author"), "url": it.get("thread_url"),
    } for it in items]
    return _frame(rows, lookback_days)


def naver(exchange, ticker, src, lookback_days):
    since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    items = run_actor(src["actor_id"], {
        "stocks": [ticker["symbol"].split(".")[0]],
        "dateFrom": since,
        "maxResults": src.get("max_items", 150),
    })
    rows = [{
        "exchange": exchange, "ticker": ticker["symbol"], "source": "naver_finance",
        "timestamp": it.get("writtenAt"),
        "text": " ".join(x for x in [it.get("title"), it.get("text")] if x),
        "likes": (it.get("likes") or 0) - (it.get("dislikes") or 0),
        "reposts": 0, "views": it.get("views") or 0, "followers": 0,
        "author": it.get("author"), "url": it.get("url"),
    } for it in items if it.get("type", "post") == "post"]
    return _frame(rows, lookback_days)


def telegram(exchange, ticker, src, lookback_days):
    key = (exchange, tuple(src["channels"]))
    if key not in _channel_cache:
        items = run_actor(src["actor_id"], {
            "channels": src["channels"],
            "lastNDays": lookback_days,
            "maxMessages": src.get("max_items", 300),
            "includeChannelInfo": True,
        })
        subs = {it.get("channelUsername") or it.get("username"): it.get("subscribers") or it.get("subscriberCount") or 0
                for it in items if it.get("rowType") == "channel"}
        _channel_cache[key] = (items, subs)
    items, subs = _channel_cache[key]
    kws = [k.lower() for k in ticker["keywords"] + ticker.get("telegram_keywords", [])]
    rows = []
    for it in items:
        if it.get("rowType") != "message":
            continue
        text = it.get("text") or ""
        if not any(k in text.lower() for k in kws):
            continue
        rows.append({
            "exchange": exchange, "ticker": ticker["symbol"], "source": "telegram",
            "timestamp": it.get("date"), "text": text,
            "likes": it.get("totalReactions") or 0,
            "reposts": 1 if it.get("isForwarded") else 0,
            "views": it.get("views") or 0,
            "followers": subs.get(it.get("channelUsername"), 0),
            "author": it.get("channelUsername"), "url": it.get("url"),
        })
    return _frame(rows, lookback_days)


ADAPTERS = {"lihkg": lihkg, "naver_finance": naver, "telegram": telegram}


def scrape(exchange, ticker, source, lookback_days):
    try:
        return ADAPTERS[source["name"]](exchange, ticker, source, lookback_days)
    except Exception as e:  # one failing source must not stop the pipeline
        print(f"  ! {source['name']} failed for {ticker['symbol']}: {e}")
        return pd.DataFrame(columns=COLS)
