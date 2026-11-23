"""Visual HTML report: tickers ranked by mentions and sentiment per exchange."""
import pandas as pd
import plotly.express as px


def build_report(summary: pd.DataFrame, path="output/report.html"):
    parts = ["<html><head><meta charset='utf-8'><title>HK & KR Sentiment Report</title></head><body>",
             "<h1>Social Sentiment Price Prediction</h1>"]
    for ex, d in summary.groupby("exchange"):
        d = d.sort_values(["mentions", "sentiment_score"], ascending=False)
        parts.append(f"<h2>{ex}</h2>")
        fig = px.scatter(d, x="sentiment_score", y="mentions", text="ticker", color="direction",
                         size=d["mentions"].clip(lower=1), title=f"{ex}: mentions vs sentiment")
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        fig2 = px.bar(d, x="ticker", y="sentiment_score", color="direction", title=f"{ex}: sentiment score")
        parts.append(fig2.to_html(full_html=False, include_plotlyjs=False))
        parts.append(d.to_html(index=False))
    parts.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path
