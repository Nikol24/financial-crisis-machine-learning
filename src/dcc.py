import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from arch import arch_model

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Stock market indices
INDEX_1 = "S&P 500"
INDEX_2 = "NIKKEI 225"


# Step 1: Load data and calculate log returns
def load_returns(filename, column_name):
    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}"
        )

    df = pd.read_excel(file_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(
        df["Close"], errors="coerce"
    )

    df = df.dropna(subset=["Date", "Close"])
    df = df[df["Close"] > 0]
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset="Date")

    df[column_name] = 100 * np.log(
        df["Close"] / df["Close"].shift(1)
    )

    return df[["Date", column_name]].dropna()


# Step 2: Load S&P 500 and NIKKEI 225
sp500 = load_returns(
    "S&P 500.xlsx", "SP500"
)

nikkei = load_returns(
    "NIKKEI 225.xlsx", "NIKKEI"
)

df = pd.merge(sp500, nikkei, on="Date")
df = df.sort_values("Date").reset_index(drop=True)

if len(df) < 30:
    raise ValueError(
        "Insufficient common observations for analysis."
    )


# Step 3: Fit GARCH(1,1) models
def get_standardized_residuals(series):
    model = arch_model(
        series,
        vol="GARCH",
        p=1,
        q=1,
        mean="Constant",
        dist="normal"
    )

    result = model.fit(disp="off")

    return (
        result.resid / result.conditional_volatility
    )


std_sp500 = get_standardized_residuals(df["SP500"])
std_nikkei = get_standardized_residuals(df["NIKKEI"])

eps = np.column_stack([
    std_sp500.to_numpy(),
    std_nikkei.to_numpy()
])

if not np.isfinite(eps).all():
    raise ValueError(
        "Non-finite standardized residuals detected."
    )


# Step 4: Calculate dynamic Q matrices
alpha = 0.01
beta = 0.98

if alpha < 0 or beta < 0 or alpha + beta >= 1:
    raise ValueError("Invalid DCC parameters.")

Q_bar = np.cov(eps, rowvar=False, ddof=1)
Q_t = Q_bar.copy()

results = []

for t in range(1, len(eps)):
    eps_previous = eps[t - 1]

    eps_outer = np.outer(
        eps_previous, eps_previous
    )

    Q_t = (
        (1 - alpha - beta) * Q_bar
        + alpha * eps_outer
        + beta * Q_t
    )

    # Convert Q_t to a correlation matrix
    diagonal = np.sqrt(np.diag(Q_t))

    R_t = Q_t / np.outer(diagonal, diagonal)

    results.append({
        "Date": df["Date"].iloc[t],
        "Q_11": Q_t[0, 0],
        "Q_12": Q_t[0, 1],
        "Q_22": Q_t[1, 1],
        "Correlation": R_t[0, 1]
    })


# Step 5: Calculate average Q matrix
dcc_df = pd.DataFrame(results)

Q_mean = pd.DataFrame(
    [
        [dcc_df["Q_11"].mean(),
         dcc_df["Q_12"].mean()],
        [dcc_df["Q_12"].mean(),
         dcc_df["Q_22"].mean()]
    ],
    index=[INDEX_1, INDEX_2],
    columns=[INDEX_1, INDEX_2]
)

print("\nAverage Dynamic Q Matrix:")
print(Q_mean.round(6))

print("\nAverage Dynamic Correlation:")
print(round(dcc_df["Correlation"].mean(), 6))


# Step 6: Save results
dcc_df.to_csv(
    RESULTS_DIR / "dcc_dynamic_results.csv",
    index=False
)

Q_mean.to_csv(
    RESULTS_DIR / "dcc_average_q_matrix.csv"
)


# Step 7: Visualize dynamic correlations
plt.figure(figsize=(12, 6))

plt.plot(
    dcc_df["Date"],
    dcc_df["Correlation"],
    label="Dynamic Correlation",
    color="blue"
)

plt.axhline(
    y=dcc_df["Correlation"].mean(),
    color="red",
    linestyle="--",
    label="Average Correlation"
)

plt.title(
    "Dynamic Conditional Correlation: "
    "S&P 500 vs NIKKEI 225"
)

plt.xlabel("Date")
plt.ylabel("Correlation")
plt.ylim(-1, 1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "dcc_correlation.png",
    dpi=300
)

plt.show()

print("\nDCC analysis completed.")
print("Results saved in the results folder.")
