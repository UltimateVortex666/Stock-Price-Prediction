# 📈 Stock Price Prediction System

A comprehensive, production-ready AI-powered stock price prediction system using deep learning neural networks (LSTM & GRU) with technical indicators. This system provides accurate predictions, backtesting, and an intuitive web interface for trading insights.

## 🎯 Features

- **Deep Learning Models**: LSTM and GRU neural networks for time-series prediction
- **Technical Indicators**: Moving averages (SMA20, SMA50) and RSI (14-period)
- **Data Preprocessing**: Normalization and time-series sequence creation
- **Model Evaluation**: RMSE (Root Mean Square Error) metrics
- **Backtesting**: Strategy performance testing against historical data
- **Professional Web Interface**: Modern, responsive UI with interactive charts
- **Extensive Dataset**: Support for 7,000+ stocks and 1,300+ ETFs

## 🏗️ Architecture

### Backend Components

1. **Data Loading** (`utils.py`): Loads stock/ETF data from local files
2. **Training** (`train.py`): Trains LSTM and GRU models with technical indicators
3. **API** (`api.py`): FastAPI server exposing prediction endpoints
4. **Frontend** (`frontend/index.html`): Professional web interface

### Model Architecture

**LSTM Model:**
- 3 LSTM layers (128 → 64 → 32 units)
- Dropout regularization (0.3, 0.3, 0.2)
- Dense layers for output

**GRU Model:**
- 3 GRU layers (128 → 64 → 32 units)
- Dropout regularization (0.3, 0.3, 0.2)
- Dense layers for output

**Features Used:**
- Close price
- SMA20 (Simple Moving Average 20-day)
- SMA50 (Simple Moving Average 50-day)
- RSI14 (Relative Strength Index 14-period)

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup Steps

1. **Clone or navigate to the project directory**

2. **Create virtual environment** (recommended):
```bash
cd backend
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Step 1: Train Models

Train models on a specific stock or ETF symbol:

```bash
cd backend
python train.py --symbol AAPL --seq_len 60 --epochs 50 --batch 64 --outdir artifacts
```

**Parameters:**
- `--symbol`: Stock/ETF symbol (e.g., AAPL, MSFT, TSLA, SPY)
- `--seq_len`: Sequence length for time-series (default: 60)
- `--epochs`: Number of training epochs (default: 10, recommended: 50+)
- `--batch`: Batch size (default: 64)
- `--outdir`: Output directory for artifacts (default: artifacts)

**Training Output:**
- `<SYMBOL>_lstm.keras` - Trained LSTM model
- `<SYMBOL>_gru.keras` - Trained GRU model
- `metrics.json` - RMSE and backtest results
- `chart.csv` - Predictions vs actual prices for visualization
- `scaler_close.json` - Data scaler for preprocessing

### Step 2: Start API Server

```bash
cd backend
uvicorn api:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### Step 3: Open Web Interface

Open `frontend/index.html` in your web browser. The interface will automatically connect to the API running on `http://localhost:8000`.

## 📡 API Endpoints

### Health Check
```
GET /health
```
Returns API health status.

### List Available Symbols
```
GET /symbols
```
Returns list of all available stock and ETF symbols from local data.

### Get Metrics
```
GET /metrics
```
Returns RMSE scores and backtest results for trained models.

### Get Chart Data
```
GET /chart?symbol=AAPL
```
Returns chart data (actual vs predicted prices) for visualization.

### Make Prediction
```
POST /predict
Content-Type: application/json

{
  "symbol": "AAPL",
  "model": "lstm",
  "seq_len": 60
}
```

Returns:
```json
{
  "symbol": "AAPL",
  "prediction_close": 185.23,
  "last_close": 182.45,
  "delta": 2.78
}
```

## 📊 Data Format

Stock/ETF data files should be in CSV format with the following columns:
- `Date` - Date of the trading day
- `Open` - Opening price
- `High` - Highest price
- `Low` - Lowest price
- `Close` - Closing price
- `Volume` - Trading volume
- `OpenInt` - Open interest (optional)

Files should be named as `<symbol>.us.txt` (e.g., `aapl.us.txt`) and placed in:
- `Data/Stocks/` for stocks
- `Data/ETFs/` for ETFs

## 🎨 Web Interface Features

### Visualization
- **Interactive Charts**: Chart.js powered visualizations showing:
  - Actual prices (white line)
  - LSTM predictions (blue dashed line)
  - GRU predictions (green dashed line)

### Prediction Interface
- Symbol search with autocomplete
- Model selection (LSTM or GRU)
- Sequence length configuration
- Real-time prediction results with:
  - Last close price
  - Predicted next close
  - Expected change (amount and percentage)

### Metrics Dashboard
- RMSE scores for both models
- Backtest results comparing:
  - Strategy returns vs Buy & Hold returns
  - Performance for both LSTM and GRU models

## 🔬 Model Evaluation

### RMSE (Root Mean Square Error)
Lower RMSE indicates better prediction accuracy. The model predicts the next day's closing price.

### Backtesting
A simple trading strategy is implemented:
- If predicted next close > current close → Go long (buy)
- Otherwise → Stay in cash

Results show:
- Strategy total return
- Buy & Hold total return
- Comparison of both approaches

## 🛠️ Technical Details

### Data Preprocessing
1. Load historical price data
2. Calculate technical indicators (SMA20, SMA50, RSI14)
3. Normalize features using MinMaxScaler
4. Create time-series sequences (sliding window)
5. Split into train (70%), validation (15%), and test (15%) sets

### Training Process
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Validation monitoring
- Model checkpointing

### Prediction Pipeline
1. Load recent data for the symbol
2. Calculate technical indicators
3. Normalize features
4. Create sequence from last N days
5. Predict using trained model
6. Inverse transform to get actual price prediction

## 📁 Project Structure

```
Stock Price Prediction/
├── backend/
│   ├── api.py              # FastAPI server
│   ├── train.py            # Model training script
│   ├── utils.py            # Data loading and utilities
│   └── requirements.txt    # Python dependencies
├── frontend/
│   └── index.html          # Web interface
├── Data/
│   ├── Stocks/             # Stock data files
│   └── ETFs/               # ETF data files
├── artifacts/              # Trained models and outputs
└── README.md               # This file
```

## 🔧 Troubleshooting

### Model Not Found Error
- Ensure you've trained models for the symbol you're trying to predict
- Check that model files exist in the `artifacts/` directory

### Data File Not Found
- Verify the symbol exists in `Data/Stocks/` or `Data/ETFs/`
- Check file naming: should be `<symbol>.us.txt` (lowercase)

### API Connection Issues
- Ensure the API server is running on port 8000
- Check CORS settings if accessing from different origin
- Verify firewall settings aren't blocking connections

### Training Issues
- Ensure sufficient data points (recommended: 1000+ days)
- Check GPU availability for faster training (optional)
- Increase epochs if models need more training

## 📈 Performance Tips

1. **More Training Data**: More historical data generally improves predictions
2. **Longer Training**: Increase epochs (50-100) for better convergence
3. **Feature Engineering**: Experiment with different technical indicators
4. **Hyperparameter Tuning**: Adjust sequence length, model architecture
5. **Ensemble Methods**: Combine LSTM and GRU predictions

## 🎓 Best Practices

1. **Train on Multiple Symbols**: Test on various stocks/ETFs
2. **Regular Retraining**: Retrain models periodically with new data
3. **Validation**: Always validate predictions on unseen data
4. **Risk Management**: Use predictions as one input, not sole decision maker
5. **Backtesting**: Always backtest strategies before live trading

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📚 References

- Long Short-Term Memory (LSTM) networks for time-series prediction
- Gated Recurrent Unit (GRU) networks
- Technical Analysis indicators (SMA, RSI)
- Backtesting methodologies for trading strategies

---

**Disclaimer**: This tool is for educational purposes only. Stock market predictions are inherently uncertain. Always do your own research and consult with financial advisors before making investment decisions.
