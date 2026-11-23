# HK & KR Social Sentiment Price Prediction Agent

Predicts each stock's next-day (n+1) close for 5 Hong Kong (HKEX) and 5 Korean (KRX) stocks from 2 social sources per exchange (via Apify) plus news (via exa.com).

## Pipeline
1. **Ingest**: Apify actors scrape the last 10 days of posts (text, UTC timestamp, likes, reposts, views, author followers). Exa returns recent news.
2. **Sentiment**: `cardiffnlp/twitter-xlm-roberta-base-sentiment` (multilingual) scores each post/article in [-1, 1]; posts are weighted by engagement.
3. **Features**: daily weighted sentiment, mentions, bullish ratio, sentiment change, plus price returns and volatility, aligned to daily UTC price bars.
4. **Model**: XGBoost on next-day return, evaluated with **walk-forward** out-of-sample testing (train on past only, predict the next day, step forward) against a no-change baseline.
5. **Output**: per ticker: name, current price, sentiment, direction, suggested position, support/resistance/pivots, predicted n+1 price, and backtest metrics. Visual report at `output/report.html`.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env   # add APIFY_TOKEN and EXA_API_KEY
python run_pipeline.py
```

## Sources and Apify actors
| Exchange | Source | Actor ID |
|---|---|---|
| HKEX | LIHKG | TODO |
| HKEX | Telegram | TODO |
| KRX | Naver Finance 종목토론실 | TODO |
| KRX | Telegram | TODO |

## Results
TODO: paste `output/summary.csv` highlights and backtest hit rate vs baseline.

## Limitations
Only ~10 days of social data exist, so sentiment features are zero before that window; results are indicative, not trading advice.
