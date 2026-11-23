import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from arch import arch_model

# Load historical stock market data
file_path = Path("data") / "^ATG_data_final.xlsx"

if not file_path.exists():
    raise FileNotFoundError(
        f"Data file not found: {file_path}"
    )

data = pd.read_excel(file_path)

# Prepare the dataset
data["Date"] = pd.to_datetime(data["Date"])
data["Close"] = pd.to_numeric(data["Close"], errors="coerce")

data = data.dropna(subset=["Date", "Close"])
data = data.sort_values("Date")
data = data[data["Close"] > 0].copy()
data = data.set_index("Date")

# Calculate daily percentage returns
data["Return"] = data["Close"].pct_change() * 100
data = data.dropna(subset=["Return"])

if len(data) < 30:
    raise ValueError("Insufficient data for GARCH analysis.")

# Fit GARCH(1,1) model
model = arch_model(
    data["Return"],
    vol="GARCH",
    p=1,
    q=1,
    mean="Constant",
    dist="normal"
)

garch_fit = model.fit(disp="off")

# Print model results
print("GARCH(1,1) Model Results")
print(garch_fit.summary())

# Forecast volatility for the next 5 trading days
forecast = garch_fit.forecast(horizon=5)

print("\nForecasted Variance (Next 5 Trading Days):")
print(forecast.variance.iloc[-1])

# Plot squared returns and estimated conditional variance
plt.figure(figsize=(12, 6))

plt.plot(
    data.index,
    data["Return"] ** 2,
    label="Actual Squared Returns",
    alpha=0.6
)

plt.plot(
    data.index,
    garch_fit.conditional_volatility ** 2,
    label="GARCH Estimated Variance",
    color="red"
)

plt.title("GARCH(1,1) Volatility Analysis - ATHEX")
plt.xlabel("Date")
plt.ylabel("Variance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
