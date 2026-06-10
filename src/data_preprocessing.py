import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """Loads the raw dataset and formats the date column."""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def handle_missing_values(df, col_threshold=0.3, row_threshold=0.5):
    """Detects and treats missing values.

    Strategy:
    - Drop features with more than col_threshold fraction missing.
    - Drop rows with more than row_threshold fraction missing.
    - Impute remaining numeric features with mean and categorical features with mode.

    This preserves as much data as possible while removing records/features with excessive missing values.
    """
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()

    if total_missing == 0:
        print("There are no missing values in the dataset.")
        return df

    print(f"Found {total_missing} missing values across {len(missing_counts[missing_counts > 0])} columns.")

    missing_ratio = missing_counts / len(df)
    cols_to_drop = missing_ratio[missing_ratio > col_threshold].index.tolist()
    if cols_to_drop:
        print(f"Dropping columns with > {col_threshold*100:.0f}% missing values: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    row_missing_ratio = df.isnull().mean(axis=1)
    rows_to_drop = row_missing_ratio[row_missing_ratio > row_threshold].index
    if len(rows_to_drop) > 0:
        print(f"Dropping {len(rows_to_drop)} rows with > {row_threshold*100:.0f}% missing values.")
        df = df.drop(index=rows_to_drop)

    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                imputed_value = df[col].mean()
                df[col] = df[col].fillna(imputed_value)
                print(f"Imputed missing values in numeric column '{col}' with mean={imputed_value:.4f}.")
            else:
                mode_values = df[col].mode(dropna=True)
                if not mode_values.empty:
                    df[col] = df[col].fillna(mode_values[0])
                    print(f"Imputed missing values in non-numeric column '{col}' with mode='{mode_values[0]}'.")
                else:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    print(f"Imputed missing values in non-numeric column '{col}' using forward/backward fill.")

    return df

def treat_outliers(df):
    """Applies IQR capping to treat outliers in numerical columns."""
    df_cleaned = df.copy()

    # Select only numerical columns
    num_cols = df_cleaned.select_dtypes(include=['number']).columns
    
    for col in num_cols:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Apply Capping
        df_cleaned[col] = np.where(df_cleaned[col] > upper_bound, upper_bound,
                                   np.where(df_cleaned[col] < lower_bound, lower_bound, df_cleaned[col]))
    
    return df_cleaned, num_cols

def scale_data(df, num_cols):
    """Applies Min-Max Scaling to numerical features."""
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[num_cols] = scaler.fit_transform(df[num_cols])
    return df_scaled

def main():
    # File paths
    input_path = '../data/raw/energy_data_set.csv'
    output_path = '../data/processed/cleaned_data.csv'
    
    print("Starting Data Preprocessing...")
    
    # Load Data
    df = load_data(input_path)
    print("Data loaded successfully.")

    # Handle Missing Values
    df = handle_missing_values(df)
    print("Missing values handled.")
    
    # Handle Outliers
    df_cleaned, num_cols = treat_outliers(df)
    print("Outliers capped using IQR method.")
    
    # Scale Data
    df_scaled = scale_data(df_cleaned, num_cols)
    print("Numerical features scaled using MinMaxScaler.")
    
    # Save Processed Data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_scaled.to_csv(output_path, index=False)
    print(f"Data preprocessing complete. Cleaned data saved to {output_path}")

if __name__ == "__main__":
    main()