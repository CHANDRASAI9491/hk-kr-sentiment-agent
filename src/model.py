"""Next-day (n+1) price prediction with walk-forward out-of-sample testing."""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

FEATURES = ["ret_1", "ret_5", "vol_5", "sent_w", "sent_mean", "mentions", "bull_ratio", "sent_chg"]


def _model():
    return XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42)


def walk_forward(df, min_train=40):
    """Train on days [0..t-1], predict day t, step forward. No look-ahead."""
    data = df.dropna(subset=FEATURES + ["target_ret"])
    preds = []
    for t in range(min_train, len(data)):
        train, test = data.iloc[:t], data.iloc[t:t + 1]
        m = _model().fit(train[FEATURES], train["target_ret"])
        p = float(m.predict(test[FEATURES])[0])
        preds.append({"date": test.index[0], "pred_ret": p, "true_ret": float(test["target_ret"].iloc[0])})
    res = pd.DataFrame(preds)
    if res.empty:
        return res, {"oos_days": 0, "direction_hit_rate": 0.0, "mae": 0.0, "naive_mae": 0.0}
    hit = (np.sign(res["pred_ret"]) == np.sign(res["true_ret"])).mean()
    mae = (res["pred_ret"] - res["true_ret"]).abs().mean()
    naive_mae = res["true_ret"].abs().mean()  # baseline: predict no change
    return res, {"oos_days": len(res), "direction_hit_rate": round(float(hit), 3),
                 "mae": round(float(mae), 4), "naive_mae": round(float(naive_mae), 4)}


def predict_next(df):
    """Fit on all labelled days, predict the next close from the latest row."""
    data = df.dropna(subset=FEATURES)
    train = data.dropna(subset=["target_ret"])
    if train.empty or data.empty:
        close = float(df["close"].iloc[-1]) if not df.empty and "close" in df else 0.0
        return {"current_price": round(close, 2), "pred_ret": 0.0, "pred_price_n1": round(close, 2), "direction": "UP"}
    m = _model().fit(train[FEATURES], train["target_ret"])
    last = data.iloc[[-1]]
    pred_ret = float(m.predict(last[FEATURES])[0])
    close = float(last["close"].iloc[0])
    return {"current_price": round(close, 2), "pred_ret": pred_ret,
            "pred_price_n1": round(close * (1 + pred_ret), 2),
            "direction": "UP" if pred_ret > 0 else "DOWN"}
