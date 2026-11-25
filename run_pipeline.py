"""End-to-end: scrape -> sentiment -> features -> predict -> report."""
import os

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.features import build_dataset, daily_sentiment
from src.model import predict_next, walk_forward
from src.news_exa import fetch_news
from src.prices import get_prices, support_resistance
from src.report import build_report
from src.scrape_social import scrape
from src.sentiment import score_texts


def main():
    load_dotenv()
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    lookback = cfg["lookback_days"]
    rows, backtests = [], []

    for ex, ex_cfg in cfg["exchanges"].items():
        for tk in ex_cfg["tickers"]:
            sym = tk["symbol"]
            print(f"[{ex}] {sym}")
            posts = pd.concat([scrape(ex, tk, src, lookback) for src in ex_cfg["sources"]], ignore_index=True)
            news = fetch_news(ex, tk, lookback)
            if (posts.empty or len(posts) == 0) and os.path.exists(f"data/{sym}_posts.csv"):
                print(f"  Using cached data/{sym}_posts.csv")
                posts = pd.read_csv(f"data/{sym}_posts.csv")
            if (news.empty or len(news) == 0) and os.path.exists(f"data/{sym}_news.csv"):
                print(f"  Using cached data/{sym}_news.csv")
                news = pd.read_csv(f"data/{sym}_news.csv")

            if not posts.empty and "sentiment" not in posts.columns:
                posts["sentiment"] = score_texts(posts["text"].tolist())
            if not news.empty and "sentiment" not in news.columns:
                news["sentiment"] = score_texts(news["text"].tolist())
            if not posts.empty:
                posts.to_csv(f"data/{sym}_posts.csv", index=False)
            if not news.empty:
                news.to_csv(f"data/{sym}_news.csv", index=False)

            prices = get_prices(sym)
            sent = daily_sentiment(posts, news)
            sent = sent[sent["ticker"] == sym].drop(columns="ticker") if not sent.empty else sent
            ds = build_dataset(prices, sent)

            bt, metrics = walk_forward(ds)
            bt["ticker"] = sym
            backtests.append(bt)
            pred = predict_next(ds)
            sr = support_resistance(prices)
            score = float(sent["sent_w"].mean()) if not sent.empty else 0.0
            rows.append({
                "exchange": ex, "ticker": sym, "name": tk["name"],
                "current_price": pred["current_price"],
                "mentions": int(len(posts) + len(news)),
                "sentiment_score": round(score, 3),
                "direction": pred["direction"],
                "pred_price_n1": pred["pred_price_n1"],
                "position": "LONG" if score > 0.15 and pred["direction"] == "UP"
                            else "SHORT" if score < -0.15 and pred["direction"] == "DOWN" else "NONE",
                **sr, **metrics,
            })

    summary = pd.DataFrame(rows)
    summary.to_csv("output/summary.csv", index=False)
    valid_bt = [b for b in backtests if not b.empty]
    if valid_bt:
        pd.concat(valid_bt).to_csv("output/backtest.csv", index=False)
    print(summary.to_string(index=False))
    print("Report:", build_report(summary))


if __name__ == "__main__":
    main()
