import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import os
import joblib

# 1. Dane treningowe — przykładowe "normalne" sekwencje
X = np.random.rand(1000, 10, 1)
y = X[:, -1, :]  # Przewidujemy ostatnią wartość sekwencji

# 2. Model LSTM
model = Sequential([
    LSTM(64, input_shape=(10, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 3. Trening
model.fit(X, y, epochs=10, validation_split=0.2, callbacks=[EarlyStopping(patience=3)])

# 4. Błąd predykcji
y_pred = model.predict(X)
errors = np.mean(np.square(y - y_pred), axis=1)

# 5. Próg błędu (np. 95 percentyl)
threshold = np.percentile(errors, 95)

# 6. Zapis modelu i progu
os.makedirs("model", exist_ok=True)
model.save("model/model.keras", save_format="keras")
joblib.dump(threshold, "model/threshold.pkl")

print("Model zapisany w 'model/model.keras'")
print("Próg błędu zapisany w 'model/threshold.pkl'")
