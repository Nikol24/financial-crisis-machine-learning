import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from arch import arch_model
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Settings
INDEX_NAME = "NIKKEI 225"
DATA_FILE = DATA_DIR / "NIKKEI 225.xlsx"

SEQUENCE_LENGTH = 10
TEST_SIZE = 30

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
df = df.set_index("Date")


# Step 2: Calculate logarithmic returns
returns = 100 * np.log(
    df["Close"] / df["Close"].shift(1)
)

returns = returns.dropna()

# Squared returns as a volatility proxy
target = returns ** 2

if len(returns) < 150:
    raise ValueError(
        "At least 150 return observations are required."
    )

split = len(returns) - TEST_SIZE

print("Dataset:", INDEX_NAME)
print("Training observations:", split)
print("Test observations:", TEST_SIZE)


# Step 3: Prepare LSTM training data
scaler = MinMaxScaler()

training_target = target.iloc[:split].to_numpy()
training_scaled = scaler.fit_transform(
    training_target.reshape(-1, 1)
)

X_train = []
y_train = []

for i in range(SEQUENCE_LENGTH, split):
    X_train.append(
        training_scaled[i-SEQUENCE_LENGTH:i]
    )
    y_train.append(training_scaled[i])

X_train = np.array(X_train)
y_train = np.array(y_train)


# Step 4: Train the LSTM model
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, 1)),
    LSTM(32),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    shuffle=False,
    verbose=1
)


# Step 5: Generate out-of-sample forecasts
garch_predictions = []
lstm_predictions = []
actual_values = []
test_dates = []

for i in range(split, len(returns)):

    # Available history before the forecast date
    history = returns.iloc[:i]

    # GARCH(1,1) one-step-ahead forecast
    garch_model = arch_model(
        history,
        vol="GARCH",
        p=1,
        q=1,
        mean="Constant",
        dist="normal"
    )

    garch_fit = garch_model.fit(disp="off")

    garch_forecast = garch_fit.forecast(
        horizon=1
    )

    garch_variance = (
        garch_forecast.variance.iloc[-1, 0]
    )

    # LSTM forecast using known past values
    previous_values = (
        target.iloc[i-SEQUENCE_LENGTH:i]
        .to_numpy()
        .reshape(-1, 1)
    )

    scaled_window = scaler.transform(
        previous_values
    )

    X_test = scaled_window.reshape(
        1, SEQUENCE_LENGTH, 1
    )

    prediction_scaled = model.predict(
        X_test, verbose=0
    )

    lstm_variance = scaler.inverse_transform(
        prediction_scaled
    )[0, 0]

    # Variance predictions cannot be negative
    lstm_variance = max(0.0, lstm_variance)

    garch_predictions.append(garch_variance)
    lstm_predictions.append(lstm_variance)

    actual_values.append(target.iloc[i])
    test_dates.append(returns.index[i])


# Step 6: Calculate forecast errors
results = pd.DataFrame({
    "Date": test_dates,
    "Actual_Squared_Return": actual_values,
    "GARCH_Forecast": garch_predictions,
    "LSTM_Forecast": lstm_predictions
})

results["GARCH_Error"] = (
    results["Actual_Squared_Return"]
    - results["GARCH_Forecast"]
)

results["LSTM_Error"] = (
    results["Actual_Squared_Return"]
    - results["LSTM_Forecast"]
)

results["GARCH_Loss"] = (
    results["GARCH_Error"] ** 2
)

results["LSTM_Loss"] = (
    results["LSTM_Error"] ** 2
)

results["Loss_Difference"] = (
    results["GARCH_Loss"]
    - results["LSTM_Loss"]
)


# Step 7: Diebold-Mariano test
# One-step forecasts: lag-0 variance estimator
d = results["Loss_Difference"].to_numpy()

T = len(d)
mean_difference = np.mean(d)
variance_difference = np.var(d, ddof=1)

if not np.isfinite(variance_difference):
    raise ValueError("Invalid loss variance.")

if variance_difference <= 0:
    raise ValueError(
        "Cannot calculate DM statistic: "
        "loss differences have zero variance."
    )

dm_statistic = mean_difference / np.sqrt(
    variance_difference / T
)

p_value = 2 * norm.sf(abs(dm_statistic))


# Step 8: Display results
print("\nDiebold-Mariano Test: GARCH vs LSTM")
print("----------------------------------")
print(f"Observations: {T}")
print(f"DM Statistic: {dm_statistic:.4f}")
print(f"P-value: {p_value:.6f}")

print(
    f"Mean GARCH Loss: "
    f"{results['GARCH_Loss'].mean():.6f}"
)

print(
    f"Mean LSTM Loss: "
    f"{results['LSTM_Loss'].mean():.6f}"
)

if p_value < 0.05:
    print(
        "Statistically significant difference "
        "at the 5% level."
    )
else:
    print(
        "No statistically significant difference "
        "at the 5% level."
    )

if mean_difference > 0:
    print("LSTM has lower average squared loss.")
elif mean_difference < 0:
    print("GARCH has lower average squared loss.")
else:
    print("Both models have equal average loss.")


# Step 9: Export results
output_file = (
    RESULTS_DIR / "garch_lstm_dm_results.csv"
)

results.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")
