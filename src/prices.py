"""Daily price bars (UTC) from Yahoo Finance."""
import pandas as pd
import yfinance as yf


def get_prices(symbol, period="6mo"):
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[cols]
    df.index = pd.to_datetime(df.index).tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df.index = df.index.normalize()
    return df.dropna(subset=["close"]) if "close" in df.columns else df


def support_resistance(prices, window=20):
    """Simple levels: recent swing low/high plus classic pivot point."""
    if prices is None or prices.empty:
        return {"support": 0.0, "resistance": 0.0, "pivot": 0.0, "s1": 0.0, "r1": 0.0}
    recent = prices.tail(window)
    last = prices.iloc[-1]
    pivot = (last["high"] + last["low"] + last["close"]) / 3
    return {
        "support": round(float(recent["low"].min()), 2),
        "resistance": round(float(recent["high"].max()), 2),
        "pivot": round(float(pivot), 2),
        "s1": round(float(2 * pivot - last["high"]), 2),
        "r1": round(float(2 * pivot - last["low"]), 2),
    }
