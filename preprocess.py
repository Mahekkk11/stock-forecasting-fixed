import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_data(df):
    """
    Basic cleaning: Drop NA, convert index to datetime
    """
    df = df.copy()
    df = df[['Close']].dropna()
    df.index = pd.to_datetime(df.index)
    return df

def normalize_data(df):
    """
    Normalize 'Close' prices using MinMaxScaler (for LSTM)
    Returns scaled data and the scaler object
    """
    df = df.copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[['Close']])
    return scaled, scaler
