import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
from metrics import calculate_metrics, store_model_score


warnings.filterwarnings("ignore")

def run_prophet(df):
    st.subheader("🔮 Prophet Forecast")

    df = df.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna()

    n_days = st.slider("Forecast days into the future:", 7, 180, 30)

    st.info("Training Prophet model...")
    model = Prophet(daily_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=n_days)
    forecast = model.predict(future)

    st.subheader("📈 Forecast Plot")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.subheader("🧩 Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    st.subheader("📄 Forecast Data")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_days))

    csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_days).to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Prophet Forecast as CSV", data=csv, file_name="prophet_forecast.csv", mime='text/csv')


    # Evaluate
    true = df['y'].iloc[-n_days:]
    pred = forecast['yhat'].iloc[-n_days:]
    rmse, mae = calculate_metrics(true, pred)
    store_model_score("Prophet", rmse, mae)
    st.success(f"📊 RMSE: {rmse}, MAE: {mae}")
