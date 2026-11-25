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
| HKEX | LIHKG | claude_code_reviewer/lihkg-scraper-en |
| HKEX | Telegram | automation-lab/telegram-scraper |
| KRX | Naver Finance 종목토론실 | hgservices/naver-stock-discussion-scraper |
| KRX | Telegram | automation-lab/telegram-scraper |

## Results
Live run on 2026-11-25 (lookback 10 days). Posts collected: LIHKG 14, Telegram (HK channels) 37, Naver 종목토론실 750, Telegram (KR channels) 326. Exa news articles: 100.

| exchange   | ticker    | name                |   current_price |   mentions |   sentiment_score | direction   |    pred_price_n1 |   direction_hit_rate |    mae |   naive_mae |
|:-----------|:----------|:--------------------|----------------:|-----------:|------------------:|:------------|-----------------:|---------------------:|-------:|------------:|
| HKEX       | 0700.HK   | Tencent             |     341.6       |         21 |             0.08  | DOWN        |    338.3         |                0.42  | 0.0195 |      0.0157 |
| HKEX       | 9988.HK   | Alibaba             |      82.2       |         30 |             0.098 | DOWN        |     81.36        |                0.481 | 0.0255 |      0.0215 |
| HKEX       | 3690.HK   | Meituan             |      64.2       |         14 |            -0.103 | DOWN        |     63.69        |                0.506 | 0.0203 |      0.018  |
| HKEX       | 1810.HK   | Xiaomi              |      26.5       |         24 |            -0.029 | DOWN        |     25.62        |                0.469 | 0.028  |      0.0235 |
| HKEX       | 0005.HK   | HSBC                |     130.6       |         12 |            -0.124 | DOWN        |    129.39        |                0.469 | 0.0146 |      0.0128 |
| KRX        | 005930.KS | Samsung Electronics |  189400         |        237 |             0.088 | UP          | 189942           |                0.423 | 0.0465 |      0.0353 |
| KRX        | 000660.KS | SK Hynix            |       1.158e+06 |        213 |             0.091 | UP          |      1.20602e+06 |                0.385 | 0.0598 |      0.0425 |
| KRX        | 035420.KS | NAVER               |  156100         |        336 |            -0.022 | DOWN        | 155301           |                0.564 | 0.0266 |      0.0219 |
| KRX        | 035720.KS | Kakao               |   30350         |        166 |            -0.089 | DOWN        |  30106.7         |                0.526 | 0.0193 |      0.0171 |
| KRX        | 005380.KS | Hyundai Motor       |  275000         |        174 |            -0.022 | UP          | 275898           |                0.5   | 0.0264 |      0.023  |

**Out-of-sample (walk-forward, ~80 trading days per stock):** mean direction hit rate 47.4%. MAE was higher than the no-change baseline for all 10 stocks, so the model does not beat a naive "tomorrow = today" forecast yet. The main reason: social and news sentiment exists only for the last 10 days, so most of the backtest runs on price features alone.

**Telegram channels used.** HK: chinadagudong, KBAsia, HANAchina, EMchina (Korean-language brokerage channels covering China/HK). KR: FastStockNews, corevalue, YeouidoStory2, stockhub_krnews. These relay news and research rather than retail chatter, so LIHKG and Naver are the true social sources and Telegram is supplementary.

**Actors:** claude_code_reviewer/lihkg-scraper-en, hgservices/naver-stock-discussion-scraper, automation-lab/telegram-scraper. News: Exa /search.

Indicative only. Not trading advice.

## Limitations
Only ~10 days of social data exist, so sentiment features are zero before that window; results are indicative, not trading advice. HK social coverage is thin (LIHKG search returned few threads in 10 days). The next step is to store posts daily so sentiment history grows and the backtest can really test it.
