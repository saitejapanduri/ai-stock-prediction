import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

from risk_analysis import calculate_risk
from news_sentiment import get_sentiment

st.set_page_config(page_title="Stock Price Prediction System")

st.title("Stock Price Prediction System")

stock = st.selectbox(
    "Select Stock",
    ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS"]
)

days = st.slider("Prediction Days", 1, 5, 3)

def build_lstm():
    model = Sequential([
        Input(shape=(60,10)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    return model

def calculate_support_resistance(df):
    support = df["Low"].rolling(20).min().iloc[-1]
    resistance = df["High"].rolling(20).max().iloc[-1]
    return support, resistance

# ---------------- NEW LOGIC FUNCTIONS ---------------- #

def calculate_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = abs(df["High"] - df["Close"].shift())
    low_close = abs(df["Low"] - df["Close"].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    atr = true_range.rolling(period).mean().iloc[-1]
    return atr


def get_entry_signal(current_price, rf_pred, lstm_pred):

    # Weighted ensemble (RF + LSTM, both in price terms)
    final_pred = (0.4 * rf_pred) + (0.6 * lstm_pred)

    if final_pred >= current_price:
        signal = "BUY"
    else:
        signal = "SELL"

    return signal, final_pred


def calculate_stoploss(entry_price, atr, signal):
    multiplier = 1.5

    if signal == "BUY":
        return entry_price - (atr * multiplier)
    elif signal == "SELL":
        return entry_price + (atr * multiplier)
    else:
        return None


def calculate_target(entry_price, stoploss, signal):
    risk = abs(entry_price - stoploss)
    reward_ratio = 2

    if signal == "BUY":
        return entry_price + (risk * reward_ratio)
    elif signal == "SELL":
        return entry_price - (risk * reward_ratio)
    else:
        return None

# -------------------------------
# LOAD MODELS
# -------------------------------
rf = pickle.load(open(f"models/{stock}_rf.pkl","rb"))
open_model = pickle.load(open(f"models/{stock}_open_model.pkl","rb"))
scaler = pickle.load(open(f"models/{stock}_scaler.pkl","rb"))

lstm = build_lstm()
lstm.load_weights(f"models/{stock}_lstm_weights.weights.h5")

# -------------------------------
# LOAD DATA
# -------------------------------
data = yf.download(stock, start="2018-01-01")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Indicators
data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()
data["NightGap"] = data["Open"] - data["Close"].shift(1)
data["Volatility"] = data["High"] - data["Low"]

delta = data["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data["RSI"] = 100 - (100 / (1 + rs))

data.dropna(inplace=True)

st.subheader("Latest Stock Data")
st.dataframe(data.tail())

# Chart
fig, ax = plt.subplots()
ax.plot(data["Close"], label="Close")
ax.plot(data["MA20"], label="MA20")
ax.plot(data["MA50"], label="MA50")
ax.legend()
st.pyplot(fig)

# Risk
risk, vol = calculate_risk(data["Close"].tail(30))
st.subheader("Risk Analysis")
st.write("Risk Level:", risk)

# Sentiment
sentiment = get_sentiment(stock)
st.subheader("Market Sentiment")
st.write("Sentiment Score:", round(sentiment,3))

if sentiment > 0.05:
    st.success("Positive Market Sentiment")
elif sentiment < -0.05:
    st.error("Negative Market Sentiment")
else:
    st.info("Neutral Market Sentiment")

# Support Resistance
support, resistance = calculate_support_resistance(data)
st.subheader("Support & Resistance")
st.write("Support Level:", round(support,2))
st.write("Resistance Level:", round(resistance,2))

# -------------------------------
# PREDICTION
# -------------------------------

if st.button("Predict Future Price"):

    predictions = []
    predicted_days = []
    trade_data = []
    temp_data = data.copy()

    for i in range(days):

        last = temp_data.iloc[-1]
        current_price = last["Close"]

        # Open prediction
        feat_open = np.array([[last["Close"], last["MA20"], last["MA50"]]])
        entry_open = float(open_model.predict(feat_open)[0])

        features = [
            entry_open,
            entry_open * 1.01,
            entry_open * 0.99,
            entry_open,
            last["Volume"],
            last["MA20"],
            last["MA50"],
            last["NightGap"],
            last["Volatility"],
            last["RSI"]
        ]

        X = np.array(features).reshape(1, -1)

        # RF predicts a RETURN -> convert to a price
        pred_rf_return = rf.predict(X)[0]
        pred_rf = current_price * (1 + pred_rf_return)

        # -------- LSTM Prediction -------- #
        scaled_data = scaler.transform(temp_data[
            ["Open","High","Low","Close","Volume","MA20","MA50","NightGap","Volatility","RSI"]
        ].values)

        X_lstm = scaled_data[-60:].reshape(1, 60, -1)
        lstm_pred_scaled = lstm.predict(X_lstm, verbose=0)[0][0]

        # LSTM predicts a SCALED ABSOLUTE Close -> inverse-transform properly
        dummy_row = np.zeros((1, len(features)))
        dummy_row[0, 3] = lstm_pred_scaled  # index 3 = "Close" column
        lstm_pred = scaler.inverse_transform(dummy_row)[0, 3]

        signal, final_pred = get_entry_signal(current_price, pred_rf, lstm_pred)

        atr = calculate_atr(temp_data)

        entry = current_price
        stoploss = calculate_stoploss(entry, atr, signal)
        target = calculate_target(entry, stoploss, signal)

        predictions.append(final_pred)

        trade_data.append([
            i+1,
            round(final_pred,2),
            signal,
            round(entry,2),
            round(stoploss,2),
            round(target,2)
        ])

        new_row = {
            "Open": entry,
            "High": max(entry, final_pred),
            "Low": min(entry, final_pred),
            "Close": final_pred,
            "Volume": last["Volume"],
            "MA20": (last["MA20"]*19 + final_pred)/20,
            "MA50": (last["MA50"]*49 + final_pred)/50,
            "NightGap": entry - last["Close"],
            "Volatility": abs(entry - final_pred),
            "RSI": last["RSI"]
        }

        temp_data = pd.concat([temp_data, pd.DataFrame([new_row])], ignore_index=True)

    future_df = pd.DataFrame(
        trade_data,
        columns=["Day","Predicted Price","Trade","Entry","StopLoss","Exit(Target)"]
    )

    st.subheader("Prediction & Trade Setup")
    st.dataframe(future_df)

    # Confidence
    volatility_penalty = min(vol / 5, 1)
    confidence = (1 - volatility_penalty) * 100

    st.write("Confidence:", round(confidence,2), "%")

    # Graph
    fig2, ax2 = plt.subplots()
    ax2.plot(range(1, days+1), predictions, marker="o", linewidth=2)
    ax2.set_title("Future Price Prediction")
    ax2.set_xlabel("Future Days")
    ax2.set_ylabel("Price (₹)")
    ax2.grid(True)
    st.pyplot(fig2)