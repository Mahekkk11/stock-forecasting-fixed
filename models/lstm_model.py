import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
from metrics import calculate_metrics, store_model_score

warnings.filterwarnings("ignore")

def run_lstm(df):
    st.subheader("🧠 LSTM Forecast (Deep Learning)")

    data = df[['Close']].dropna()
    data.index = pd.to_datetime(data.index)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    window = 60
    X, y = [], []
    for i in range(window, len(scaled_data)):
        X.append(scaled_data[i-window:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    st.info("Training LSTM model (may take a few seconds)...")
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    n_days = st.slider("Forecast days into the future:", 7, 60, 30)
    input_seq = scaled_data[-window:]
    predictions = []

    for _ in range(n_days):
        pred_input = input_seq[-window:]
        pred_input = pred_input.reshape(1, window, 1)
        pred = model.predict(pred_input, verbose=0)
        predictions.append(pred[0, 0])
        input_seq = np.append(input_seq, pred)[-window:]

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': predicted_prices.flatten()})
    forecast_df.set_index('Date', inplace=True)

    st.subheader("📈 LSTM Forecast Plot")
    fig, ax = plt.subplots()
    data['Close'].plot(ax=ax, label='Historical')
    forecast_df['Forecast'].plot(ax=ax, label='LSTM Forecast', color='orange')
    ax.legend()
    st.pyplot(fig)

    st.subheader("📄 Forecast Data")
    st.dataframe(forecast_df.tail())

    csv = forecast_df.to_csv().encode('utf-8')
    st.download_button("📥 Download LSTM Forecast as CSV", data=csv, file_name="lstm_forecast.csv", mime='text/csv')

    

    # Evaluate
    true = data['Close'].iloc[-n_days:]
    pred = forecast_df['Forecast'].iloc[:len(true)]
    rmse, mae = calculate_metrics(true, pred)
    store_model_score("LSTM", rmse, mae)
    st.success(f"📊 RMSE: {rmse}, MAE: {mae}")
