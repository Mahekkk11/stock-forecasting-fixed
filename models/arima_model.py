import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings
from metrics import calculate_metrics, store_model_score


warnings.filterwarnings("ignore")

def run_arima(df):
    st.subheader("🔍 ARIMA Forecast")
    df = df[['Close']].dropna()
    n_days = st.slider("Forecast days into the future:", 7, 90, 30)

    st.info("Training ARIMA model...")
    model = ARIMA(df['Close'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=n_days)

    forecast_index = pd.date_range(start=df.index[-1], periods=n_days+1, freq='D')[1:]
    forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_index)

    fig, ax = plt.subplots()
    df['Close'].plot(ax=ax, label="Historical")
    forecast_df['Forecast'].plot(ax=ax, label="ARIMA Forecast", color='red')
    ax.legend()
    st.pyplot(fig)

    st.write("📈 Forecast Data")
    st.dataframe(forecast_df)

    csv = forecast_df.to_csv().encode('utf-8')
    st.download_button("📥 Download ARIMA Forecast as CSV", data=csv, file_name="arima_forecast.csv", mime='text/csv')


    # Evaluate
    true = df['Close'].iloc[-n_days:]
    pred = forecast[:len(true)]
    rmse, mae = calculate_metrics(true, pred)
    store_model_score("ARIMA", rmse, mae)
    st.success(f"📊 RMSE: {rmse}, MAE: {mae}")
