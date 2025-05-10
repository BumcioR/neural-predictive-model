# neural-predictive-model

This project is designed to detect anomalies in sequential data using an LSTM model. 
It learns patterns of typical behavior and flags input data as anomalies if they significantly deviate.

## Clone the repository:
git clone <URL_REPO>
cd <PROJECT>
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate.bat      # Windows
pip install -r requirements.txt

## Run training script:
python train/train_model.py

This will save the trained model to model/model.keras and the error threshold to model/threshold.pkl.

## Start FastAPI server:
uvicorn app.main:app --reload

## Open REST API documentation at:
http://127.0.0.1:8000/docs

## Use /predict endpoint and pass sequence of values in JSON format

# The service runs locally at localhost:8000.
Calling /predict endpoint accurately identifies normal and anomalous data sequences.