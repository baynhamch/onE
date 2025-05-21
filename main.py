import yfinance as yf
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, jsonify

app = Flask(__name__)
trade_signals = {}

company_names = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "GOOG": "Alphabet Inc.", "NVDA": "NVIDIA Corp.",
    "AMZN": "Amazon.com Inc.", "META": "Meta Platforms", "TSLA": "Tesla Inc.", "AMD": "Advanced Micro Devices",
    "CRM": "Salesforce", "NFLX": "Netflix", "PLTR": "Palantir", "SMCI": "Supermicro", "AVGO": "Broadcom",
    "INTC": "Intel", "MU": "Micron", "ARM": "Arm Holdings", "QCOM": "Qualcomm", "ASML": "ASML Holding",
    "AI": "C3.ai", "SOUN": "SoundHound", "RIVN": "Rivian", "LCID": "Lucid", "ARKK": "ARK Innovation ETF",
    "ARKW": "ARK Next Gen ETF", "ROKU": "Roku", "SHOP": "Shopify", "COIN": "Coinbase", "SPOT": "Spotify",
    "UPST": "Upstart", "U": "Unity Software", "JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "V": "Visa", "MA": "Mastercard", "PYPL": "PayPal", "AXP": "AmEx", "DFS": "Discover", "SOFI": "SoFi",
    "UNH": "UnitedHealth", "PFE": "Pfizer", "MRNA": "Moderna", "CVS": "CVS Health", "LLY": "Eli Lilly",
    "HIMS": "Hims & Hers", "VRTX": "Vertex", "ISRG": "Intuitive Surgical", "REGN": "Regeneron", "BMY": "Bristol-Myers"
}

def fetch_stock_data(stocks):
    df = yf.download(
        tickers=stocks,
        period="5d",
        interval="5m",
        group_by='ticker',
        auto_adjust=True,
        threads=True
    )
    return df

def calculate_indicators(df):
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df.dropna()

def generate_signals(df):
    df = df.copy()
    df['Buy_Signal'] = (df['SMA_20'] > df['SMA_50']) & (df['RSI'] < 30)
    df['Sell_Signal'] = (df['SMA_20'] < df['SMA_50']) & (df['RSI'] > 70)
    return df

def process_stocks(df, stocks):
    processed = {}
    for stock in stocks:
        try:
            stock_df = pd.DataFrame(df[stock]['Close']).rename(columns={'Close': 'Close'})
            if stock_df.empty:
                continue
            processed[stock] = generate_signals(calculate_indicators(stock_df))
        except Exception as e:
            print(f"❌ {stock}: {e}")
    return processed

def train_ml_model(all_data):
    X = all_data[['SMA_20', 'SMA_50', 'RSI']].dropna()
    y = all_data['Buy_Signal'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"✅ Model Accuracy: {model.score(X_test, y_test):.2f}")
    return model, X

def predict_trades(model, X, stocks, processed_data):
    latest_data = X.tail(len(processed_data))
    predicted = model.predict(latest_data)

    signals = {}
    for stock, pred in zip(processed_data, predicted):
        row = processed_data[stock].iloc[-1]
        price = row['Close']
        timestamp = processed_data[stock].index[-1]
        name = company_names.get(stock, stock)

        if pred == 1:
            signal = "BUY"
            conf = round(model.predict_proba([row[['SMA_20', 'SMA_50', 'RSI']]])[0][1], 2)
            reason = f"SMA_20 ({row['SMA_20']:.2f}) > SMA_50 ({row['SMA_50']:.2f}) and RSI ({row['RSI']:.2f}) < 30"
        elif row['Sell_Signal']:
            signal = "SELL"
            conf = 0.9
            reason = f"SMA_20 ({row['SMA_20']:.2f}) < SMA_50 ({row['SMA_50']:.2f}) and RSI ({row['RSI']:.2f}) > 70"
        else:
            signal = "HOLD"
            conf = 0.5
            reason = f"SMA_20 ({row['SMA_20']:.2f}) <= SMA_50 ({row['SMA_50']:.2f}) or RSI ({row['RSI']:.2f}) neutral"

        signals[stock] = {
            "ticker": stock,
            "name": name,
            "signal": signal,
            "confidence": conf,
            "reason": reason,
            "price": round(price, 2),
            "timestamp": timestamp.isoformat()
        }

    return signals

@app.route("/signal/<ticker>")
def get_signal(ticker):
    ticker = ticker.upper()
    if ticker in trade_signals:
        print(f"📤 Sending to app: {trade_signals[ticker]}")
        return jsonify(trade_signals[ticker])
    return jsonify({"error": "Ticker not found"}), 404

def load_signals():
    global trade_signals
    tickers = list(company_names.keys())
    df = fetch_stock_data(tickers)
    processed = process_stocks(df, tickers)
    all_data = pd.concat(processed.values(), keys=processed.keys())
    model, X = train_ml_model(all_data)
    trade_signals = predict_trades(model, X, tickers, processed)
    print(f"✅ Loaded {len(trade_signals)} signals.")


if __name__ == "__main__":
    load_signals()
    app.run(debug=True, port=5001, host="0.0.0.0", threaded=True)