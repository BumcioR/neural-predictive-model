
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np

app = FastAPI()

# Wczytaj model
model = load_model("model/model.keras")

# Schemat danych wejściowych
class PredictionRequest(BaseModel):
    data: list[list[float]]  # np. [[0.1], [0.2], ..., [0.9]]

# Schemat odpowiedzi
class PredictionResponse(BaseModel):
    prediction: float

# Endpoint startowy przekierowujący do dokumentacji
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

# Główny endpoint predykcyjny
@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    input_array = np.array(req.data).reshape(1, len(req.data), 1)
    prediction = model.predict(input_array)[0][0]
    return PredictionResponse(prediction=float(prediction))
