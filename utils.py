import os
import glob
import json
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def add_indicators(df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
    df = df.copy()
    if close_col not in df.columns:
        # try common variants
        for cand in ["Close", "Adj Close", "adj_close", "Adj_Close"]:
            if cand in df.columns:
                close_col = cand
                break
    df["SMA20"] = df[close_col].rolling(20).mean()
    df["SMA50"] = df[close_col].rolling(50).mean()
    df["RSI14"] = rsi(df[close_col], 14)
    df = df.dropna().reset_index(drop=True)
    return df

def infer_schema_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize common column names
    cols = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    for want in ["date","open","high","low","close","volume","symbol","name","ticker"]:
        for c in df.columns:
            lc = c.lower()
            if lc == want or (want=="symbol" and lc in ["symbol","name","ticker"]) or (want=="date" and "date" in lc):
                mapping[want] = c
    # Build standardized df
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[mapping.get("date", list(df.columns)[0])])
    out["open"] = pd.to_numeric(df.get(mapping.get("open","open"), df.select_dtypes(include='number').iloc[:,0]), errors='coerce')
    out["high"] = pd.to_numeric(df.get(mapping.get("high","high"), df.select_dtypes(include='number').iloc[:,0]), errors='coerce')
    out["low"] = pd.to_numeric(df.get(mapping.get("low","low"), df.select_dtypes(include='number').iloc[:,0]), errors='coerce')
    # try close
    close_source = mapping.get("close", None)
    if close_source is None:
        for cand in ["close","adj close","adj_close","adjclose"]:
            for c in df.columns:
                if c.lower() == cand:
                    close_source = c
                    break
            if close_source: break
    if close_source is None:
        # fallback: last numeric column
        close_source = df.select_dtypes(include='number').columns[-1]
    out["close"] = pd.to_numeric(df[close_source], errors='coerce')
    out["volume"] = pd.to_numeric(df.get(mapping.get("volume","volume"), 0), errors='coerce')
    sym_col = mapping.get("symbol", None)
    if sym_col is None:
        out["symbol"] = "UNKNOWN"
    else:
        out["symbol"] = df[sym_col].astype(str)
    out = out.sort_values("date").dropna().reset_index(drop=True)
    return out

def _resolve_data_dir(data_dir: str | None = None) -> str:
    """Return absolute path to the Data directory.
    Priority: provided data_dir (absolute) > provided (relative to project root) > default ../Data from this file.
    Supports env var DATA_DIR to override.
    """
    env_dir = os.environ.get("DATA_DIR")
    if env_dir:
        return os.path.abspath(env_dir)
    if data_dir:
        # If absolute, return; else resolve relative to project root (../ from backend)
        if os.path.isabs(data_dir):
            return data_dir
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.abspath(os.path.join(base, data_dir))
    # default: ../Data relative to this file
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))


def load_local_stock_data(target_symbol: str = "AAPL", data_dir: str = "Data") -> pd.DataFrame:
    """
    Loads stock/ETF data from local Data/Stocks and Data/ETFs folders.
    Returns a standardized dataframe filtered to target_symbol.
    """
    # Resolve data directory to absolute path
    data_root = _resolve_data_dir(data_dir)
    # Try to find the file in Stocks first, then ETFs
    symbol_upper = target_symbol.upper()
    
    # Try Stocks folder
    stocks_path = os.path.join(data_root, "Stocks", f"{symbol_upper.lower()}.us.txt")
    etfs_path = os.path.join(data_root, "ETFs", f"{symbol_upper.lower()}.us.txt")
    
    file_path = None
    if os.path.exists(stocks_path):
        file_path = stocks_path
    elif os.path.exists(etfs_path):
        file_path = etfs_path
    else:
        # Try alternative naming (e.g., aapl.us.txt vs AAPL.us.txt)
        stocks_files = glob.glob(os.path.join(data_root, "Stocks", f"*{symbol_upper.lower()}*.txt"))
        etfs_files = glob.glob(os.path.join(data_root, "ETFs", f"*{symbol_upper.lower()}*.txt"))
        all_files = stocks_files + etfs_files
        if all_files:
            file_path = all_files[0]
        else:
            raise FileNotFoundError(f"Could not find data file for symbol {target_symbol} in {data_root}")
    
    # Read the file
    df = pd.read_csv(file_path)
    df = infer_schema_and_standardize(df)
    
    # Add symbol column if not present
    if "symbol" not in df.columns or df["symbol"].isna().all():
        df["symbol"] = symbol_upper
    
    # Add indicators
    data = add_indicators(df, "close")
    return data

def load_kaggle_prices(target_symbol: str = "AAPL") -> pd.DataFrame:
    """
    Deprecated Kaggle loader: now always loads from local Data/Stocks and Data/ETFs.
    Kept for backward compatibility with existing imports.
    """
    return load_local_stock_data(target_symbol)

def list_available_symbols(data_dir: str = "Data") -> Dict[str, List[str]]:
    """
    Lists all available stock and ETF symbols from local data folders.
    Returns dict with 'stocks' and 'etfs' keys.
    """
    root = _resolve_data_dir(data_dir)
    stocks_dir = os.path.join(root, "Stocks")
    etfs_dir = os.path.join(root, "ETFs")
    
    stocks = []
    etfs = []
    
    if os.path.exists(stocks_dir):
        stock_files = glob.glob(os.path.join(stocks_dir, "*.txt"))
        for f in stock_files:
            base = os.path.basename(f)
            symbol = base.replace(".us.txt", "").upper()
            if symbol:
                stocks.append(symbol)
    
    if os.path.exists(etfs_dir):
        etf_files = glob.glob(os.path.join(etfs_dir, "*.txt"))
        for f in etf_files:
            base = os.path.basename(f)
            symbol = base.replace(".us.txt", "").upper()
            if symbol:
                etfs.append(symbol)
    
    return {"stocks": sorted(stocks), "etfs": sorted(etfs)}

def make_sequences(values: np.ndarray, seq_len: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(seq_len, len(values)):
        X.append(values[i-seq_len:i])
        y.append(values[i])
    return np.array(X), np.array(y)

def train_val_test_split(X, y, train=0.7, val=0.15):
    n = len(X)
    n_train = int(n*train)
    n_val = int(n*val)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
