from pydantic import BaseModel
from typing import List

class AnomalyRequest(BaseModel):
    data: List[List[float]]  # np. [[0.1], [0.2], ..., [1.0]]

class AnomalyResponse(BaseModel):
    prediction: float
    error: float
    is_anomaly: bool

