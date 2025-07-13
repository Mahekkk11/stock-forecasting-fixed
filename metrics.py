from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

model_scores = {}  # Global dictionary to store model results

def calculate_metrics(true_values, predicted_values):
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    return round(rmse, 2), round(mae, 2)

def store_model_score(name, rmse, mae):
    model_scores[name] = {"RMSE": rmse, "MAE": mae}

def get_all_scores():
    return model_scores
