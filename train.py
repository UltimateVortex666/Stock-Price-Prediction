import os, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

try:
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    import keras
    from keras import layers

from utils import load_local_stock_data, make_sequences, train_val_test_split

import argparse

def build_lstm(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_gru(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.GRU(64),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def backtest(prices: np.ndarray, preds: np.ndarray) -> dict:
    # simple strategy: if predicted next close > today close => long next day else cash
    # compute returns vs buy & hold
    # align lengths
    p = prices[-len(preds):]
    # daily returns
    ret = np.diff(p) / p[:-1]
    # signal for day t is based on prediction for t (predicting price at t) compared to price t-1
    signal = preds[:-1] > p[:-1]
    strat_ret = ret * signal
    equity = (1 + strat_ret).cumprod()
    bh_equity = (1 + ret).cumprod()
    return {
        "strategy_total_return": float(equity[-1] - 1),
        "buyhold_total_return": float(bh_equity[-1] - 1),
        "final_equity_strategy": float(equity[-1]),
        "final_equity_buyhold": float(bh_equity[-1])
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--seq_len", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--outdir", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_local_stock_data(args.symbol)
    # features: [close, SMA20, SMA50, RSI14]
    feats = df[["close","SMA20","SMA50","RSI14"]].values.astype(np.float32)
    scaler = MinMaxScaler()
    feats_scaled = scaler.fit_transform(feats)

    X, y = make_sequences(feats_scaled[:,0], seq_len=args.seq_len)  # predict normalized close
    # expand feature dimension: use all features in each timestep
    # Build sequences for all features to feed model
    X_all = []
    for i in range(args.seq_len, len(feats_scaled)):
        X_all.append(feats_scaled[i-args.seq_len:i, :])
    X_all = np.array(X_all)
    y = feats_scaled[args.seq_len:, 0:1]  # next normalized close
    (Xtr, ytr), (Xv, yv), (Xte, yte) = train_val_test_split(X_all, y)

    input_shape = (Xtr.shape[1], Xtr.shape[2])

    lstm = build_lstm(input_shape)
    gru = build_gru(input_shape)

    cb = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")]

    hist_lstm = lstm.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=args.epochs, batch_size=args.batch, callbacks=cb, verbose=2)
    hist_gru = gru.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=args.epochs, batch_size=args.batch, callbacks=cb, verbose=2)

    # predictions on test
    pred_lstm = lstm.predict(Xte).reshape(-1,1)
    pred_gru = gru.predict(Xte).reshape(-1,1)

    # invert scale for RMSE
    def invert_close(y_scaled):
        tmp = np.zeros((len(y_scaled), feats.shape[1]), dtype=np.float32)
        tmp[:,0] = y_scaled.reshape(-1)
        inv = scaler.inverse_transform(tmp)[:,0]
        return inv

    yte_inv = invert_close(yte)
    lstm_inv = invert_close(pred_lstm)
    gru_inv = invert_close(pred_gru)

    rmse_lstm = sqrt(mean_squared_error(yte_inv, lstm_inv))
    rmse_gru = sqrt(mean_squared_error(yte_inv, gru_inv))

    # backtest on the last test window using model predictions aligned with actual prices
    # reconstruct prices aligned to test set
    close_series = df["close"].values.astype(np.float32)
    test_prices = close_series[-len(yte_inv):]
    bt_lstm = backtest(test_prices, lstm_inv)
    bt_gru = backtest(test_prices, gru_inv)

    # save artifacts
    lstm.save(os.path.join(args.outdir, f"{args.symbol}_lstm.keras"))
    gru.save(os.path.join(args.outdir, f"{args.symbol}_gru.keras"))
    with open(os.path.join(args.outdir, "scaler_close.json"), "w") as f:
        json.dump({"min_": scaler.min_.tolist(), "scale_": scaler.scale_.tolist(), "n_features_in_": int(scaler.n_features_in_)}, f)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({
            "symbol": args.symbol,
            "rmse": {"lstm": rmse_lstm, "gru": rmse_gru},
            "backtest": {"lstm": bt_lstm, "gru": bt_gru}
        }, f, indent=2)

    # also dump a small CSV for the frontend chart
    chart = pd.DataFrame({
        "date": df["date"].iloc[-len(yte_inv):].values,
        "actual_close": yte_inv,
        "lstm_pred": lstm_inv,
        "gru_pred": gru_inv
    })
    chart.to_csv(os.path.join(args.outdir, "chart.csv"), index=False)
    chart.to_csv(os.path.join(args.outdir, f"{args.symbol}_chart.csv"), index=False)

    print("Training complete. Artifacts in", args.outdir)

if __name__ == "__main__":
    main()
