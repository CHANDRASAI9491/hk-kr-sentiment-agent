"""Recent news per ticker through exa.com."""
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from exa_py import Exa


def fetch_news(exchange, ticker, lookback_days, num_results=10):
    exa = Exa(os.environ["EXA_API_KEY"])
    since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    query = f"{ticker['name']} {ticker['symbol']} stock news"
    res = exa.search_and_contents(
        query, num_results=num_results, start_published_date=since, text={"max_characters": 1500}
    )
    rows = [{
        "exchange": exchange,
        "ticker": ticker["symbol"],
        "source": "exa_news",
        "timestamp": r.published_date,
        "title": r.title,
        "url": r.url,
        "text": f"{r.title or ''}. {(r.text or '')[:1500]}",
    } for r in res.results]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df
