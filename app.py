import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

from metrics import get_all_scores
from models import arima_model, prophet_model, lstm_model
from login import show_login, show_user_header

# Set page config
st.set_page_config(page_title="📈 Stock Forecast", layout="wide")

# Authentication
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    show_login()
    st.stop()

# Show user info
show_user_header()

# Apply custom CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ styles.css not found. Using default theme.")

local_css("styles.css")

# Ticker input
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, TSLA, INFY):", "INFY.NS")

# Download data with fallback
try:
    df = yf.download(ticker, start='2015-01-01', end='2024-12-31')
    if df.empty:
        raise ValueError(f"⚠️ No data found for '{ticker}'. Please try another valid symbol.")
    df.index = pd.to_datetime(df.index)
except Exception as e:
    st.warning(f"⚠️ Online fetch failed: {e}. Trying fallback data...")
    try:
        df = pd.read_csv("fallback_data.csv", index_col=0)
        df.index = pd.to_datetime(df.index)
        st.success("✅ Loaded fallback stock data.")
    except FileNotFoundError:
        st.error("❌ No fallback data found. Please upload 'fallback_data.csv'.")
        st.stop()

# Show raw data
st.subheader(f"📄 Raw Data for {ticker}")
st.dataframe(df.tail())

# Plot closing prices
st.subheader("📉 Closing Price Trend")
fig, ax = plt.subplots()
df['Close'].plot(ax=ax, title=f"{ticker} - Closing Prices", color='blue')
ax.set_ylabel("Price")
st.pyplot(fig)

# Model selection
st.subheader("🔮 Choose a Forecasting Model")
model_choice = st.selectbox("Select Model", ["None", "ARIMA", "Prophet", "LSTM"])

if model_choice == "ARIMA":
    with st.expander("🔍 ARIMA Forecast"):
        arima_model.run_arima(df)

elif model_choice == "Prophet":
    with st.expander("🔮 Prophet Forecast"):
        prophet_model.run_prophet(df)

elif model_choice == "LSTM":
    with st.expander("🧠 LSTM Forecast"):
        lstm_model.run_lstm(df)

# Model performance
st.subheader("📊 Model Performance Comparison")
scores = get_all_scores()

if scores:
    result_df = pd.DataFrame(scores).T
    st.dataframe(result_df)

    rmse_fig = px.bar(
        result_df.reset_index(), x='index', y='RMSE',
        color='index',
        color_discrete_map={'ARIMA': 'red', 'Prophet': 'green', 'LSTM': 'orange'},
        title='RMSE Comparison'
    )
    st.plotly_chart(rmse_fig)

    mae_fig = px.bar(
        result_df.reset_index(), x='index', y='MAE',
        color='index',
        color_discrete_map={'ARIMA': 'red', 'Prophet': 'green', 'LSTM': 'orange'},
        title='MAE Comparison'
    )
    st.plotly_chart(mae_fig)
else:
    st.info("ℹ️ Run at least one model to see comparison.")

# Extra visualizations
with st.expander("📊 Extra Data Visualizations"):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.map(lambda x: x[0])

    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    st.write("### Moving Averages (30 & 100 days)")
    fig_ma, ax_ma = plt.subplots()
    df['Close'].plot(ax=ax_ma, label=ticker, color='blue')
    df['MA30'].plot(ax=ax_ma, label='30-Day MA', linestyle='--')
    df['MA100'].plot(ax=ax_ma, label='100-Day MA', linestyle='--', color='orange')
    ax_ma.legend()
    st.pyplot(fig_ma)

    st.write("### Trading Volume")
    fig_vol, ax_vol = plt.subplots()
    df['Volume'].plot(ax=ax_vol, color='purple')
    ax_vol.set_ylabel("Volume")
    st.pyplot(fig_vol)

    st.write("### Volume vs Close Price")
    fig_vol_price = px.scatter(df, x='Volume', y='Close', title="Volume vs Close Price", color='Close')
    st.plotly_chart(fig_vol_price)

    df['Daily Return'] = df['Close'].pct_change()
    st.write("### Daily Returns")
    st.line_chart(df['Daily Return'])

    df['Cumulative Return'] = (1 + df['Daily Return']).cumprod()
    st.write("### Cumulative Return")
    st.line_chart(df['Cumulative Return'])

    df['Rolling Volatility'] = df['Daily Return'].rolling(window=30).std()
    st.write("### Rolling 30-Day Volatility")
    st.line_chart(df['Rolling Volatility'])

    st.write("### Histogram of Daily Returns")
    fig_hist, ax_hist = plt.subplots()
    df['Daily Return'].hist(bins=50, ax=ax_hist, color='skyblue')
    ax_hist.set_title("Histogram of Daily Returns")
    st.pyplot(fig_hist)

    st.write("### 💹 Candlestick Chart")
    candlestick = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )
    ])
    candlestick.update_layout(title=f"{ticker} - Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(candlestick)

    # 📆 Monthly Average Close Price
st.write("### 📆 Monthly Average Close Price")

try:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index is not DateTime. Cannot resample.")

    monthly_avg = df['Close'].resample('M').mean()
    monthly_df = pd.DataFrame({'Month': monthly_avg.index, 'AvgClose': monthly_avg.values})
    fig_month = px.bar(monthly_df, x='Month', y='AvgClose', title="Monthly Average Close Price", color='AvgClose')
    fig_month.update_traces(marker_color='teal')
    st.plotly_chart(fig_month)

except Exception as e:
    st.warning(f"⚠️ Skipping Monthly Average Chart due to: {e}")





# # 📈 All Combined Forecasts
# if st.checkbox("📈 Show All Forecast Charts Together"):
#     st.subheader("📊 All Forecasts Combined")
#     fig, ax = plt.subplots()
#     df['Close'].plot(ax=ax, label="📘 Historical")

#     # 🔴 ARIMA
#     if "ARIMA" in scores:
#         from statsmodels.tsa.arima.model import ARIMA
#         model = ARIMA(df['Close'], order=(5, 1, 0)).fit()
#         forecast = model.forecast(steps=30)
#         forecast_index = pd.date_range(start=df.index[-1], periods=31, freq='D')[1:]
#         ax.plot(forecast_index, forecast, label="🔴 ARIMA", color='red')

#     # 🟢 Prophet
#     if "Prophet" in scores:
#         from prophet import Prophet
#         try:
#             temp_df = df.reset_index()
#             # Rename columns to match Prophet expectations
#             if 'Date' in temp_df.columns:
#                 temp_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
#             elif 'index' in temp_df.columns:
#                 temp_df.rename(columns={'index': 'ds', 'Close': 'y'}, inplace=True)
#             else:
#                 temp_df.rename(columns={temp_df.columns[0]: 'ds', 'Close': 'y'}, inplace=True)

#             temp_df['ds'] = pd.to_datetime(temp_df['ds'], errors='coerce')
#             temp_df['y'] = pd.to_numeric(temp_df['y'], errors='coerce')
#             temp_df = temp_df[['ds', 'y']].dropna()

#             if not temp_df.empty and temp_df['y'].ndim == 1:
#                 model = Prophet(daily_seasonality=True)
#                 model.fit(temp_df)
#                 future = model.make_future_dataframe(periods=30)
#                 forecast = model.predict(future)
#                 ax.plot(forecast['ds'].iloc[-30:], forecast['yhat'].iloc[-30:], label="🟢 Prophet", color='green')
#             else:
#                 st.warning("⚠️ Prophet skipped: 'y' column is not 1D or data is empty.")
#         except Exception as e:
#             st.warning(f"⚠️ Prophet forecast skipped due to error: {e}")

#     # 🟠 LSTM
#     if "LSTM" in scores:
#         try:
#             from sklearn.preprocessing import MinMaxScaler
#             from tensorflow.keras.models import Sequential
#             from tensorflow.keras.layers import LSTM, Dense
#             import numpy as np

#             close_data = df[['Close']].dropna()

#             if len(close_data) > 100:
#                 scaler = MinMaxScaler()
#                 scaled = scaler.fit_transform(close_data)
#                 X, y = [], []
#                 window = 60
#                 for i in range(window, len(scaled)):
#                     X.append(scaled[i-window:i, 0])
#                     y.append(scaled[i, 0])
#                 X = np.array(X)
#                 y = np.array(y)

#                 if X.ndim == 2:
#                     X = X.reshape((X.shape[0], X.shape[1], 1))

#                 model = Sequential()
#                 model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
#                 model.add(LSTM(50))
#                 model.add(Dense(1))
#                 model.compile(optimizer='adam', loss='mean_squared_error')
#                 model.fit(X, y, epochs=5, batch_size=32, verbose=0)

#                 input_seq = scaled[-window:]
#                 lstm_preds = []
#                 for _ in range(30):
#                     pred_input = input_seq[-window:]
#                     pred_input = pred_input.reshape(1, window, 1)
#                     pred = model.predict(pred_input, verbose=0)
#                     lstm_preds.append(pred[0, 0])
#                     input_seq = np.append(input_seq, pred)[-window:]

#                 lstm_preds = scaler.inverse_transform(np.array(lstm_preds).reshape(-1, 1))
#                 future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)
#                 ax.plot(future_dates, lstm_preds, label="🟠 LSTM", color='orange')
#             else:
#                 st.warning("⚠️ Not enough data to train LSTM model.")
#         except Exception as e:
#             st.warning(f"⚠️ LSTM forecast skipped due to error: {e}")

#     ax.set_title("📊 Combined Forecasts")
#     ax.set_ylabel("Price")
#     ax.legend()
#     st.pyplot(fig)
