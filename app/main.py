from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

with open("files/arima_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Homepage"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/predict")
def predict(steps: int):
    forecast = model.forecast(steps=steps)
    return {"forecast": forecast.tolist()}


