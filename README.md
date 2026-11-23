# Financial Crisis Analysis & Prediction Using Machine Learning

**MSc Thesis | Financial Data Science | Machine Learning | Time Series Analysis**

**Author:** Nikoleta Kostea  
**University:** Neapolis University Pafos  
**Degree:** MSc in Data Analytics and Financial Technology  
**Year:** 2025  
**Programming Language:** Python

---

## Project Overview

This project is based on my completed Master's thesis, which investigates the impact of the **2007–2008 Global Financial Crisis** and the **COVID-19 pandemic** on six major international stock markets.

The research examines financial market volatility, dynamic correlations, forecasting performance, and the potential application of Machine Learning to financial risk monitoring.

Using historical financial data covering **2003–2023**, the study combines traditional econometric methods with advanced machine learning techniques.

The methods include:

- GARCH(1,1) — Financial volatility modeling
- Dynamic Conditional Correlation (DCC) — Cross-market dependence analysis
- Diebold–Mariano Test — Statistical comparison of forecasting performance
- Long Short-Term Memory (LSTM) — Neural network forecasting

The project demonstrates the application of Python, financial econometrics, statistical analysis, and deep learning to real-world financial data.

## Research Objectives

The primary objectives of this research are:

1. Analyze financial market behavior before, during, and after major global crises.
2. Examine volatility patterns across international stock markets.
3. Investigate dynamic correlations and financial market interdependence.
4. Compare the forecasting performance of statistical and Machine Learning models.
5. Explore the potential of LSTM neural networks in financial forecasting and early warning applications.

## Financial Markets Analyzed

The research focuses on six international stock market indices.

| Country | Stock Market Index |
|---|---|
| Greece | ATHEX |
| France | CAC 40 |
| Germany | DAX 40 |
| United Kingdom | FTSE 100 |
| United States | S&P 500 |
| Japan | NIKKEI 225 |

**Data source:** Yahoo Finance

**Research period:** January 2003 – December 2023

## Historical Periods

To investigate market behavior under different economic conditions, the research divides the historical dataset into six periods.

| Period | Date Range |
|---|---|
| Before the Global Financial Crisis | January 2003 – June 2007 |
| During the Global Financial Crisis | July 2007 – June 2009 |
| After the Global Financial Crisis | July 2009 – December 2015 |
| Before COVID-19 | January 2016 – December 2019 |
| During COVID-19 | January 2020 – June 2021 |
| After COVID-19 | July 2021 – December 2023 |

## Methodology

### 1. GARCH(1,1) — Volatility Analysis

The Generalized Autoregressive Conditional Heteroskedasticity model is used to analyze changes in financial market volatility.

The model investigates volatility clustering and the persistence of financial market uncertainty across different economic periods.

**Python implementation:** `src/garch.py`

### 2. Dynamic Conditional Correlation (DCC)

Dynamic Conditional Correlation analysis is used to investigate time-varying relationships between international financial markets.

The Python implementation calculates dynamic correlation matrices using standardized residuals from individual GARCH(1,1) models.

The current implementation uses fixed DCC recursion parameters rather than independently estimating them through maximum likelihood.

**Python implementation:** `src/dcc.py`

### 3. Diebold–Mariano Test

The Diebold–Mariano statistical test is applied to compare forecasting performance between competing predictive models.

The GitHub implementation compares GARCH(1,1) and LSTM volatility forecasts using squared forecast errors and an estimate of the long-run variance of loss differences.

**Python implementation:** `src/diebold_mariano.py`

### 4. Long Short-Term Memory (LSTM)

LSTM neural networks are designed to identify temporal relationships within sequential data.

The original research explores LSTM forecasting and potential applications to financial early warning systems.

The Python portfolio implementation additionally examines S&P 500 closing-price prediction, using chronological training and testing periods.

Model performance is assessed using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Comparison against a naive forecasting baseline

**Python implementation:** `src/lstm.py`

## Technologies & Tools

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| Financial Econometrics | ARCH |
| Statistical Analysis | SciPy |
| Data Files | Excel, OpenPyXL |
| Version Control | Git, GitHub |

## Repository Structure

```text
financial-crisis-machine-learning/
|
|-- data/
|   |-- ATHEX.xlsx
|   |-- CAC 40.xlsx
|   |-- DAX 40.xlsx
|   |-- FTSE 100.xlsx
|   |-- S&P 500.xlsx
|   |-- NIKKEI 225.xlsx
|   |-- README.md
|
|-- src/
|   |-- garch.py
|   |-- dcc.py
|   |-- diebold_mariano.py
|   |-- lstm.py
|   |-- README.md
|
|-- .gitignore
|-- requirements.txt
|-- README.md
```

The Python scripts can generate output files and visualizations in a `results/` directory.

## Installation

Clone the repository:

```bash
git clone https://github.com/Nikol24/financial-crisis-machine-learning.git
```

Navigate to the project directory:

```bash
cd financial-crisis-machine-learning
```

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## Running the Models

Run the GARCH volatility analysis:

```bash
python src/garch.py
```

Run the DCC analysis:

```bash
python src/dcc.py
```

Run the Diebold–Mariano forecasting comparison:

```bash
python src/diebold_mariano.py
```

Run the LSTM forecasting model:

```bash
python src/lstm.py
```

The scripts require historical price datasets with `Date` and `Close` columns. Successful execution and reproducibility of the portfolio implementations should be verified separately from the original thesis results.

## Research Findings

The completed Master's thesis provides a comparative investigation of international financial market behavior during two major periods of global economic instability.

The analysis examines how market volatility and cross-market dependence vary across countries and economic conditions.

The research also investigates the use of statistical forecasting methods and LSTM neural networks for financial time series analysis.

The findings provide a basis for understanding international market relationships, volatility dynamics, and the potential role of quantitative models in financial risk assessment.

The original thesis findings and the subsequent GitHub code implementations are distinct components of this project.

## Research Significance

This project combines financial research with practical programming and data science techniques.

It demonstrates experience in:

- Financial time series analysis
- Statistical and econometric modeling
- Deep learning with LSTM networks
- Data preprocessing and feature engineering
- Forecast evaluation and model comparison
- Financial data visualization
- Python-based analytical workflows

## Limitations

Financial markets are influenced by unpredictable economic, political, and global events. Forecasting models cannot guarantee accurate future predictions or reliably identify every financial crisis.

The original thesis and the subsequent GitHub implementations are not identical in every modeling procedure. Results from the portfolio code should therefore be evaluated independently.

The analyzed stock market indices are broad market indicators rather than direct measures of financial-sector-only performance.

This project is intended for academic research, education, and demonstration of analytical methods. It does not constitute investment advice.

## Author

**Nikoleta Kostea**

MSc in Data Analytics and Financial Technology  
Neapolis University Pafos

GitHub: [Nikol24](https://github.com/Nikol24)

**Research completed:** June 2025
