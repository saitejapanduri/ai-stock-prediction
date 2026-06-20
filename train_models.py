import os
import yfinance as yf
import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

stocks = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS"]

LOOKBACK = 60

os.makedirs("models", exist_ok=True)

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 3])  # Close price
    return np.array(X), np.array(y)

for stock in stocks:

    print("Training", stock)

    df = yf.download(stock, start="2018-01-01")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Indicators
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["NightGap"] = df["Open"] - df["Close"].shift(1)
    df["Volatility"] = df["High"] - df["Low"]

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)

    # Target
    df["Return"] = df["Close"].pct_change().shift(-1)

    # -------------------------------
    # ✅ OPEN PRICE MODEL (LR)
    # -------------------------------
    X_open = df[["Close","MA20","MA50"]].iloc[:-1].values
    y_open = df["Open"].shift(-1).dropna().values

    open_model = LinearRegression()
    open_model.fit(X_open, y_open)

    pickle.dump(open_model, open(f"models/{stock}_open_model.pkl","wb"))

    # -------------------------------
    # ✅ RANDOM FOREST
    # -------------------------------
    features = [
        "Open","High","Low","Close","Volume",
        "MA20","MA50","NightGap","Volatility","RSI"
    ]

    X = df[features].iloc[:-1]
    y_return = df["Return"].iloc[:-1]

    rf = RandomForestRegressor(n_estimators=200)
    rf.fit(X, y_return)

    pickle.dump(rf, open(f"models/{stock}_rf.pkl","wb"))

    # -------------------------------
    # ✅ LSTM
    # -------------------------------
    data = df[features].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    pickle.dump(scaler, open(f"models/{stock}_scaler.pkl","wb"))

    X_lstm, y_lstm = create_sequences(scaled, LOOKBACK)

    model = Sequential([
        Input(shape=(LOOKBACK,len(features))),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_lstm, y_lstm, epochs=5, batch_size=32, verbose=0)

    model.save_weights(f"models/{stock}_lstm_weights.weights.h5")

print("All models trained successfully")