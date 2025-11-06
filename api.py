import os, json
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import MinMaxScaler

# Fix TensorFlow/Keras import compatibility
try:
    from tensorflow import keras
except ImportError:
    import keras

# Import utils (without error for missing custom functions)
try:
    from utils import add_indicators, list_available_symbols, load_local_stock_data
except ImportError:
    from utils import add_indicators
    def list_available_symbols():
        return ["Function not implemented."]


app = FastAPI(title="Stocks LSTM/GRU Predictor", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories for saved models and frontend
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
FRONTEND_DIR = os.environ.get(
    "FRONTEND_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
)

# ✅ Serve frontend at / and /web
if os.path.isdir(FRONTEND_DIR):
    app.mount("/web", StaticFiles(directory=FRONTEND_DIR, html=True), name="web")


@app.get("/")
def root_index():
    """Serve the frontend index.html"""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "Frontend not found. Open /web or check your folder.",
        "FRONTEND_DIR": FRONTEND_DIR
    }


class PredictPayload(BaseModel):
    symbol: str = "AAPL"
    seq_len: int = 60
    model: str = "lstm"  # or "gru"


def load_scaler(path: str, n_features: int):
    with open(path, "r") as f:
        obj = json.load(f)
    scaler = MinMaxScaler()
    scaler.min_ = np.array(obj["min_"])
    scaler.scale_ = np.array(obj["scale_"])
    scaler.n_features_in_ = n_features
    return scaler


@app.get("/health")
def health():
    return {"status": "API is running ✅"}


@app.get("/symbols")
def symbols():
    """List available stock symbols."""
    try:
        return list_available_symbols()
    except Exception as e:
        return {"error": str(e)}


@app.get("/data-check")
def data_check(symbol: str = "AAPL"):
    """Diagnose data loading paths for a symbol."""
    info = {"symbol": symbol.upper()}
    try:
        df = load_local_stock_data(symbol)
        info["local_rows"] = int(len(df))
        info["status"] = "local_ok"
    except Exception as e:
        info["local_error"] = str(e)
        info["status"] = "local_failed"
    return info


@app.get("/metrics")
def metrics():
    """Return training metrics (RMSE, Backtest)."""
    path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    if not os.path.exists(path):
        return {"error": "No metrics found. Please train your model first."}
    with open(path) as f:
        return json.load(f)


@app.get("/chart")
def chart(symbol: str = Query(None)):
    """Return prediction & actual CSV data for chart rendering."""
    if symbol:
        path = os.path.join(ARTIFACTS_DIR, f"{symbol.upper()}_chart.csv")
        if not os.path.exists(path):
            path = os.path.join(ARTIFACTS_DIR, "chart.csv")
    else:
        path = os.path.join(ARTIFACTS_DIR, "chart.csv")

    if not os.path.exists(path):
        return {"error": "Chart data not found. Train the model first."}

    df = pd.read_csv(path)
    return {
        "date": df["date"].tolist(),
        "actual": df["actual_close"].tolist(),
        "lstm": df["lstm_pred"].tolist(),
        "gru": df["gru_pred"].tolist(),
    }


@app.post("/predict")
def predict(payload: PredictPayload):
    """Return next-day predicted stock price."""
    sym = payload.symbol.upper()
    seq_len = payload.seq_len
    model_name = payload.model.lower()

    # Load & preprocess data (local only)
    try:
        df = load_local_stock_data(sym)
    except Exception as e:
        return {
            "error": f"Could not load data for {sym}: {e}",
            "hint": "Ensure Data/Stocks/<symbol>.us.txt or set DATA_DIR to your Data folder."
        }
    feats = df[["close","SMA20","SMA50","RSI14"]].values.astype("float32")
    scaler = MinMaxScaler()
    feats_scaled = scaler.fit_transform(feats)

    X = np.array([feats_scaled[-seq_len:]])

    # Load trained model
    model_path = os.path.join(ARTIFACTS_DIR, f"{sym}_{model_name}.keras")
    if not os.path.exists(model_path):
        # fallback to generic if symbol-specific not available
        model_path = os.path.join(ARTIFACTS_DIR, f"{model_name}.keras")
    if not os.path.exists(model_path):
        return {"error":"model not found; please run training"}
    model = keras.models.load_model(model_path)
    y_scaled = model.predict(X).reshape(-1,1)

    # invert
    tmp = np.zeros((len(y_scaled), feats.shape[1]), dtype=np.float32)
    tmp[:,0] = y_scaled.reshape(-1)
    inv = scaler.inverse_transform(tmp)[:,0]

    return {
        "symbol": sym,
        "prediction_close": float(inv[-1]),
        "last_close": float(df["close"].iloc[-1]),
        "delta": float(inv[-1] - df["close"].iloc[-1]),
    }


if __name__ == "__main__":
    try:
        import uvicorn
        # Note: reload is only supported when running via CLI import string (e.g., `uvicorn api:app --reload`).
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except ModuleNotFoundError:
        print("uvicorn is not installed. Install with: pip install uvicorn[standard]")
