import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle


def train_arima_model(data, order=(5, 1, 0)):
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    train = pd.to_numeric(train, errors="coerce")
    test = pd.to_numeric(test, errors="coerce")

    model = ARIMA(train, order=order)
    model_fit = model.fit()

    predictions = model_fit.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mae = mean_absolute_error(test, predictions)

    return model_fit, rmse, mae


def log_modeling_with_mlflows(data_path, order):
    
    df = pd.read_csv(data_path, skiprows=3, names=["Date", "Close"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    data = df["Close"].values
    
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("ARIMA_Time_Series")
    with mlflow.start_run():
        model, rmse, mae = train_arima_model(data, order=order)

        mlflow.log_param("order", order)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        
        model_path = "arima_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(model_path)
        
        input_example = np.array(data[:10])
        mlflow.log_dict({"input_example": input_example.tolist()}, "input_example.json")

    print("MLflow logged")


if __name__ == "__main__":

    df = pd.read_csv("stock_data.csv", skiprows=3, names=["Date", "Close"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    data = df["Close"].values

    model, rmse, mae = train_arima_model(data)
    print(f"RMSE: {rmse}, MAE: {mae}")

    with open("arima_model.pkl", "wb") as f:
        pickle.dump(model, f)

    log_modeling_with_mlflows("stock_data.csv", order=(5, 1, 0))
