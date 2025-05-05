from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    values: List[float]  # np. 10 punktów czasu
