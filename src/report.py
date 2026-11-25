"""Visual HTML report: tickers ranked by mentions and sentiment per exchange."""
import os
import shutil
import pandas as pd
import plotly.express as px


def build_report(summary: pd.DataFrame, path="output/report.html"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    total_mentions = int(summary["mentions"].sum()) if "mentions" in summary else 0
    avg_hit_rate = summary["direction_hit_rate"].mean() if "direction_hit_rate" in summary else 0.0
    up_count = int((summary["direction"] == "UP").sum()) if "direction" in summary else 0
    down_count = int((summary["direction"] == "DOWN").sum()) if "direction" in summary else 0
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HK & KR Social Sentiment Price Prediction Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --surface: #1e293b;
      --surface-border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --green: #22c55e;
      --green-bg: rgba(34, 197, 94, 0.15);
      --red: #ef4444;
      --red-bg: rgba(239, 68, 68, 0.15);
      --badge-none: rgba(148, 163, 184, 0.2);
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 2rem;
    }}
    .container {{
      max-width: 1280px;
      margin: 0 auto;
    }}
    header {{
      margin-bottom: 2rem;
      border-bottom: 1px solid var(--surface-border);
      padding-bottom: 1.5rem;
    }}
    h1 {{
      font-size: 2rem;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }}
    h1 span {{
      color: var(--primary);
    }}
    p.subtitle {{
      color: var(--text-muted);
      margin-top: 0.5rem;
      font-size: 0.95rem;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1.25rem;
      margin-bottom: 2.5rem;
    }}
    .stat-card {{
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 12px;
      padding: 1.25rem;
    }}
    .stat-label {{
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 0.5rem;
    }}
    .stat-value {{
      font-size: 1.85rem;
      font-weight: 700;
      color: #fff;
    }}
    .stat-sub {{
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 0.25rem;
    }}
    .exchange-section {{
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 14px;
      padding: 1.75rem;
      margin-bottom: 2.5rem;
    }}
    .exchange-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.5rem;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid var(--surface-border);
    }}
    .exchange-title {{
      font-size: 1.4rem;
      font-weight: 600;
    }}
    .charts-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
      margin-bottom: 2rem;
    }}
    @media (max-width: 900px) {{
      .charts-grid {{ grid-template-columns: 1fr; }}
    }}
    .chart-box {{
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--surface-border);
      border-radius: 10px;
      padding: 1rem;
    }}
    .table-container {{
      overflow-x: auto;
      border-radius: 10px;
      border: 1px solid var(--surface-border);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 0.9rem;
    }}
    th {{
      background: #0f172a;
      color: var(--text-muted);
      font-weight: 600;
      padding: 0.85rem 1rem;
      border-bottom: 1px solid var(--surface-border);
      white-space: nowrap;
    }}
    td {{
      padding: 0.85rem 1rem;
      border-bottom: 1px solid rgba(51, 65, 85, 0.4);
      white-space: nowrap;
    }}
    tr:hover td {{
      background: rgba(255, 255, 255, 0.02);
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.25rem 0.65rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .badge-up {{
      color: var(--green);
      background: var(--green-bg);
      border: 1px solid rgba(34, 197, 94, 0.3);
    }}
    .badge-down {{
      color: var(--red);
      background: var(--red-bg);
      border: 1px solid rgba(239, 68, 68, 0.3);
    }}
    .badge-long {{
      color: var(--green);
      background: var(--green-bg);
    }}
    .badge-short {{
      color: var(--red);
      background: var(--red-bg);
    }}
    .badge-none {{
      color: var(--text-muted);
      background: var(--badge-none);
    }}
    .ticker-symbol {{
      font-weight: 700;
      color: var(--primary);
    }}
    .ticker-name {{
      color: var(--text);
    }}
    footer {{
      margin-top: 3rem;
      text-align: center;
      color: var(--text-muted);
      font-size: 0.85rem;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1><span>HK & KR</span> Social Sentiment & Price Prediction</h1>
      <p class="subtitle">Multi-source NLP sentiment analysis (LIHKG, Naver, Telegram) combined with XGBoost walk-forward price predictions</p>
    </header>

    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">Coverage</div>
        <div class="stat-value">{len(summary)} Tickers</div>
        <div class="stat-sub">5 HKEX + 5 KRX equities</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Total Mentions Scraped</div>
        <div class="stat-value">{total_mentions:,}</div>
        <div class="stat-sub">Social discussions & news articles</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Walk-Forward Hit Rate</div>
        <div class="stat-value">{avg_hit_rate:.1%}</div>
        <div class="stat-sub">Out-of-sample directional accuracy</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Market Bias</div>
        <div class="stat-value"><span style="color:var(--green)">{up_count} ▲</span> / <span style="color:var(--red)">{down_count} ▼</span></div>
        <div class="stat-sub">Next-day predicted direction</div>
      </div>
    </div>
"""

    theme_layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(15, 23, 42, 0.4)",
        "font": {"color": "#94a3b8", "family": "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto"},
        "margin": {"l": 50, "r": 30, "t": 50, "b": 40},
        "xaxis": {"gridcolor": "#334155", "zerolinecolor": "#475569"},
        "yaxis": {"gridcolor": "#334155", "zerolinecolor": "#475569"},
    }

    for ex, d in summary.groupby("exchange"):
        d = d.sort_values(["mentions", "sentiment_score"], ascending=False)
        
        # Mentions vs Sentiment bubble chart
        fig1 = px.scatter(
            d,
            x="sentiment_score",
            y="mentions",
            text="ticker",
            color="direction",
            size=d["mentions"].clip(lower=2),
            color_discrete_map={"UP": "#22c55e", "DOWN": "#ef4444"},
            title=f"{ex}: Mentions vs. Sentiment Score",
        )
        fig1.update_layout(**theme_layout)
        fig1.update_traces(textposition="top center")
        
        # Sentiment bar chart
        fig2 = px.bar(
            d,
            x="ticker",
            y="sentiment_score",
            color="direction",
            color_discrete_map={"UP": "#22c55e", "DOWN": "#ef4444"},
            title=f"{ex}: Net Social Sentiment Score",
        )
        fig2.update_layout(**theme_layout)
        
        chart1_div = fig1.to_html(full_html=False, include_plotlyjs=False)
        chart2_div = fig2.to_html(full_html=False, include_plotlyjs=False)
        
        rows_html = []
        for _, row in d.iterrows():
            dir_badge = f'<span class="badge badge-up">▲ UP</span>' if row["direction"] == "UP" else f'<span class="badge badge-down">▼ DOWN</span>'
            pos = str(row.get("position", "NONE")).upper()
            pos_class = "badge-long" if pos == "LONG" else "badge-short" if pos == "SHORT" else "badge-none"
            pos_badge = f'<span class="badge {pos_class}">{pos}</span>'
            
            curr_p = f"{row['current_price']:,.2f}" if isinstance(row['current_price'], (int, float)) else str(row['current_price'])
            pred_p = f"{row['pred_price_n1']:,.2f}" if isinstance(row['pred_price_n1'], (int, float)) else str(row['pred_price_n1'])
            hit_rate = f"{row['direction_hit_rate']:.1%}" if isinstance(row.get('direction_hit_rate'), (int, float)) else "-"
            
            rows_html.append(f"""
            <tr>
              <td><span class="ticker-symbol">{row['ticker']}</span></td>
              <td class="ticker-name">{row['name']}</td>
              <td><strong>{curr_p}</strong></td>
              <td>{row['mentions']}</td>
              <td>{row['sentiment_score']:+.3f}</td>
              <td>{dir_badge}</td>
              <td><strong>{pred_p}</strong></td>
              <td>{pos_badge}</td>
              <td>{row.get('pivot', '-')}</td>
              <td>{row.get('s1', '-')} / {row.get('support', '-')}</td>
              <td>{row.get('r1', '-')} / {row.get('resistance', '-')}</td>
              <td>{hit_rate}</td>
              <td>{row.get('mae', '-')}</td>
            </tr>
            """)
        
        tbody_content = "\n".join(rows_html)
        
        html += f"""
    <div class="exchange-section">
      <div class="exchange-header">
        <h2 class="exchange-title">{ex} Markets</h2>
        <span class="stat-sub">{len(d)} Equities Tracked</span>
      </div>

      <div class="charts-grid">
        <div class="chart-box">{chart1_div}</div>
        <div class="chart-box">{chart2_div}</div>
      </div>

      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Company</th>
              <th>Current Price</th>
              <th>Mentions</th>
              <th>Sentiment</th>
              <th>Pred Direction</th>
              <th>Pred Price (n+1)</th>
              <th>Signal</th>
              <th>Pivot</th>
              <th>Support (S1 / Low)</th>
              <th>Resistance (R1 / High)</th>
              <th>Hit Rate</th>
              <th>MAE</th>
            </tr>
          </thead>
          <tbody>
            {tbody_content}
          </tbody>
        </table>
      </div>
    </div>
"""

    html += """
    <footer>
      <p>Data ingested via Apify actors (LIHKG, Naver 종목토론실, Telegram) and Exa. Multilingual scoring via XLM-RoBERTa. Walk-forward backtests evaluated out-of-sample.</p>
      <p style="margin-top:0.25rem; font-size:0.75rem; color:#64748b;">Indicative analytics only. Not financial or trading advice.</p>
    </footer>
  </div>
</body>
</html>
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
        
    # Also write or copy to root report.html if different
    root_report = "report.html"
    try:
        if os.path.abspath(path) != os.path.abspath(root_report):
            shutil.copy2(path, root_report)
    except Exception:
        pass

    return path
