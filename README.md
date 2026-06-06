# Appliance Energy Prediction

## Overview
This repository contains an end-to-end machine learning pipeline designed to predict appliance energy consumption using multivariate time-series data.

The final architecture utilizes a **PyTorch Long Short-Term Memory (LSTM)** neural network, benchmarked against standard Machine Learning algorithms (Random Forest and Linear Regression) to ensure optimal predictive performance.

---

## Repository Structure

```text
├── data/
│   ├── raw/                   # Original dataset (energy_data_set.csv)
│   └── processed/             # Cleaned and engineered datasets
├── notebooks/
│   └── EDA.ipynb              # Exploratory Data Analysis and visual insights
├── src/
│   ├── data_preprocessing.py  # Handles missing values, IQR capping, and scaling
│   ├── feature_engineering.py # Generates time-lags, rolling means, and RF feature selection
│   ├── model.py               # PyTorch LSTM and baseline model architectures
│   └── train.py               # Temporal splitting, training loops, and evaluation
├── models/
│   └── optimized_lstm.pth     # Saved PyTorch model weights
├── reports/
│   └── Report.pdf             # Comprehensive methodology and findings
├── requirements.txt           # Project dependencies
└── README.md                  # Setup and execution instructions
```

## Setup Instructions

1. Environment Configuration
It is highly recommended to use a virtual environment to prevent dependency conflicts.

Using venv:
```
# Create the virtual environment
python -m venv venv

# Activate the environment (Windows)
venv\Scripts\activate

# Activate the environment (Mac/Linux)
source venv/bin/activate
```
2. Install Dependencies
Ensure you are in the root directory of the project, then run:

```
pip install -r requirements.txt
```

## How to Run the Code
The pipeline is designed to be highly modular. To reproduce the results, execute the scripts in the following order from the root directory:

Step 1: Data Preprocessing
Cleans the raw data, applies IQR capping for outliers, and scales numerical features.

```
python src/data_preprocessing.py
```
Step 2: Feature Engineering
Creates temporal features, 10m/30m/1h lag indicators, rolling averages, and executes Random Forest feature selection to retain only the top 15 most predictive columns.

```
python src/feature_engineering.py
```
Step 3: Model Training and Evaluation
Splits the data chronologically (80% Train / 20% Test) to prevent temporal leakage, trains the baseline models, trains the PyTorch LSTM, and outputs the MAE/RMSE benchmark scores.

```
python src/train.py
```

