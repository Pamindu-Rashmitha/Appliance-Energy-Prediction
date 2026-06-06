import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """Loads the raw dataset and formats the date column."""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
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