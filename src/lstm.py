import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model settings
DATA_FILE = DATA_DIR / "S&P 500.xlsx"
LOOKBACK = 30
TRAIN_RATIO = 0.8

np.random.seed(42)
tf.random.set_seed(42)


# Step 1: Load historical data
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Data file not found: {DATA_FILE}"
    )

df = pd.read_excel(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"])
df["Close"] = pd.to_numeric(
    df["Close"], errors="coerce"
)

df = df.dropna(subset=["Date", "Close"])
df = df[df["Close"] > 0]
df = df.sort_values("Date")
df = df.drop_duplicates(subset="Date")
df = df.reset_index(drop=True)

data = df[["Close"]].values

if len(data) < 100:
    raise ValueError("Insufficient historical observations.")


# Step 2: Chronological train-test split
split = int(len(data) * TRAIN_RATIO)

train_data = data[:split]

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data)

scaled_data = scaler.transform(data)


# Step 3: Create sequences
X_train, y_train = [], []
X_test, y_test = [], []

for i in range(LOOKBACK, len(scaled_data)):
    X_window = scaled_data[i-LOOKBACK:i]
    y_value = scaled_data[i]

    if i < split:
        X_train.append(X_window)
        y_train.append(y_value)
    else:
        X_test.append(X_window)
        y_test.append(y_value)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)


# Step 4: Build LSTM model
model = Sequential([
    Input(shape=(LOOKBACK, 1)),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(25),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


# Step 5: Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    shuffle=False,
    verbose=1
)


# Step 6: Predict test observations
predicted_scaled = model.predict(
    X_test, verbose=0
)

predictions = scaler.inverse_transform(
    predicted_scaled
)

actual = scaler.inverse_transform(y_test)


# Step 7: Evaluate model performance
mae = mean_absolute_error(
    actual, predictions
)

rmse = np.sqrt(
    mean_squared_error(actual, predictions)
)

# Naive baseline: tomorrow equals today's close
naive_predictions = data[split-1:-1]

naive_mae = mean_absolute_error(
    actual, naive_predictions
)

print("\nLSTM Forecasting Results")
print("------------------------")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Naive Baseline MAE: {naive_mae:.4f}")


# Step 8: Save predictions
test_dates = df["Date"].iloc[split:]

results = pd.DataFrame({
    "Date": test_dates.values,
    "Actual": actual.flatten(),
    "LSTM_Prediction": predictions.flatten(),
    "Naive_Prediction": naive_predictions.flatten()
})

results.to_csv(
    RESULTS_DIR / "lstm_predictions.csv",
    index=False
)


# Step 9: Visualize predictions
plt.figure(figsize=(12, 6))

plt.plot(
    results["Date"],
    results["Actual"],
    label="Actual S&P 500"
)

plt.plot(
    results["Date"],
    results["LSTM_Prediction"],
    label="LSTM Predictions",
    color="red"
)

plt.title("S&P 500: Actual vs LSTM Predictions")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "lstm_forecast.png",
    dpi=300
)

plt.show()


# Step 10: Forecast the next trading observation
last_sequence = scaled_data[-LOOKBACK:]

X_future = last_sequence.reshape(
    1, LOOKBACK, 1
)

future_scaled = model.predict(
    X_future, verbose=0
)

future_prediction = scaler.inverse_transform(
    future_scaled
)

print(
    "\nNext trading observation forecast: "
    f"{future_prediction[0, 0]:.2f}"
)

print("\nLSTM analysis completed.")
