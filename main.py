import yfinance as yf
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Function to fetch stock data
def fetch_stock_data(stocks):
    """Download the most recent intraday stock data for multiple stocks."""
    df = yf.download(
        tickers=stocks,
        period="5d",     # Last 5 days
        interval="5m",   # 5-minute intervals (most granular Yahoo allows for 5d)
        group_by='ticker',
        auto_adjust=True,
        threads=True
    )
    return df


# Function to calculate indicators
def calculate_indicators(df):
    """Calculate SMA and RSI for a given stock dataframe."""
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df.dropna()


# Function to generate trading signals
def generate_signals(df):
    """Generate Buy/Sell signals."""
    df = df.copy()
    df['Buy_Signal'] = (df['SMA_20'] > df['SMA_50']) & (df['RSI'] < 30)
    df['Sell_Signal'] = (df['SMA_20'] < df['SMA_50']) & (df['RSI'] > 70)
    return df


# Function to process stocks
def process_stocks(df, stocks):
    """Apply indicators & generate signals for each stock."""
    processed_data = {}
    for stock in stocks:
        stock_df = pd.DataFrame(df[stock]['Close']).rename(columns={'Close': 'Close'})
        processed_data[stock] = generate_signals(calculate_indicators(stock_df))
    return processed_data


# Function to train ML model
def train_ml_model(all_data):
    """Train a Random Forest model for stock predictions."""
    X = all_data[['SMA_20', 'SMA_50', 'RSI']].dropna()
    y = all_data['Buy_Signal'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"✅ Model Accuracy: {accuracy:.2f}")

    return model, X


# Function to predict buy/sell signals
def predict_trades(model, X, stocks, processed_data):
    """Predict Buy/Sell/Hold signals for each stock and provide reasoning."""
    latest_data = X.tail(len(stocks))
    predicted_signals = model.predict(latest_data)

    trade_signals = {}
    for stock, signal in zip(stocks, predicted_signals):
        stock_df = processed_data[stock]
        stock_data = stock_df.iloc[-1]
        timestamp = stock_df.index[-1]
        price = stock_data['Close']

        if signal == 1:
            reason = f"SMA_20 ({stock_data['SMA_20']:.2f}) > SMA_50 ({stock_data['SMA_50']:.2f}) and RSI ({stock_data['RSI']:.2f}) < 30 (oversold)"
            trade_signals[stock] = f"📈 BUY Signal for {stock} @ ${price:.2f} on {timestamp} — {reason}"
        elif stock_data['Sell_Signal']:
            reason = f"SMA_20 ({stock_data['SMA_20']:.2f}) < SMA_50 ({stock_data['SMA_50']:.2f}) and RSI ({stock_data['RSI']:.2f}) > 70 (overbought)"
            trade_signals[stock] = f"🔻 SELL Signal for {stock} @ ${price:.2f} on {timestamp} — {reason}"
        else:
            reason = f"SMA_20 ({stock_data['SMA_20']:.2f}) <= SMA_50 ({stock_data['SMA_50']:.2f}) or RSI ({stock_data['RSI']:.2f}) in neutral range"
            trade_signals[stock] = f"⏸️ HOLD for {stock} @ ${price:.2f} on {timestamp} — {reason}"

    return trade_signals


# Main function
def main():
    stocks = ["SPY", "AAPL", "TSLA", "NVDA", "META", "SPY", "GOOG", "NVDA", "MCD"]

    df = fetch_stock_data(stocks)
    processed_data = process_stocks(df, stocks)

    all_data = pd.concat(processed_data.values(), keys=processed_data.keys())
    model, X = train_ml_model(all_data)

    trade_signals = predict_trades(model, X, stocks, processed_data)

    for stock, decision in trade_signals.items():
        print(decision)


# Run the script
if __name__ == "__main__":
    main()