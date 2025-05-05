# train/train_model.py
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Dane przykładowe
X = np.random.rand(1000, 10, 1)
y = np.random.rand(1000, 1)

model = Sequential([
    LSTM(64, input_shape=(10, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, validation_split=0.2, callbacks=[EarlyStopping(patience=3)])

os.makedirs("model", exist_ok=True)
model.save("model/model.keras", save_format="keras")
print("Model saved to model/model.keras")
