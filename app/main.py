from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from keras.models import load_model
import joblib
import numpy as np
from .schemas import AnomalyRequest, AnomalyResponse

app = FastAPI()

# Wczytanie modelu i progu
model = load_model("model/model.keras")
threshold = joblib.load("model/threshold.pkl")

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.post("/predict", response_model=AnomalyResponse)
def predict(req: AnomalyRequest):
    input_array = np.array(req.data).reshape(1, len(req.data), 1)
    prediction = model.predict(input_array)[0][0]
    true_value = input_array[0, -1, 0]
    error = (true_value - prediction) ** 2
    is_anomaly = error > threshold
    return AnomalyResponse(
        prediction=float(prediction),
        error=float(error),
        is_anomaly=bool(is_anomaly)
    )
