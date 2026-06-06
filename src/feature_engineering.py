import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor

def load_processed_data(file_path):
    """Loads the processed dataset for feature engineering."""
    df = pd.read_csv(file_path, parse_dates=['date'])
    return df

def create_time_features(df):
    """Extracts hour, day, month, and weekend indicators."""
    df_feat = df.copy()
    df_feat['hour'] = df_feat['date'].dt.hour
    df_feat['day_of_week'] = df_feat['date'].dt.dayofweek
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['is_weekend'] = df_feat['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    return df_feat

def create_rolling_features(df, target_col='Appliances', windows=[6, 18]):
    """
    Creates rolling mean features.
    windows=[6, 18] corresponds to 1-hour and 3-hour averages.
    """
    df_feat = df.copy()
    for window in windows:
        df_feat[f'{target_col}_rolling_mean_{window}'] = df_feat[target_col].rolling(window=window).mean()
    return df_feat

def create_lagged_features(df, target_col='Appliances', lags=[1, 3, 6]):
    """
    Creates lag features for previous time steps.
    lags=[1, 3, 6] corresponds to 10 mins, 30 mins, and 1 hour ago.
    """
    df_feat = df.copy()
    for lag in lags:
        df_feat[f'{target_col}_lag_{lag}'] = df_feat[target_col].shift(lag)
    return df_feat

def create_interaction_features(df):
    """Creates interaction terms between key environmental features."""
    df_feat = df.copy()
    if 'T_out' in df_feat.columns and 'RH_out' in df_feat.columns:
        df_feat['T_out_RH_out_interact'] = df_feat['T_out'] * df_feat['RH_out']
    return df_feat

def select_top_features(df, target_col='Appliances', top_n=15):
    """
    Uses a Random Forest Regressor to determine feature importance 
    and filters the dataset down to the top N features.
    """
    print("Evaluating feature importance...")
    # Drop rows with NaNs created by lagging/rolling
    df_clean = df.dropna().copy()
    
    # Separate features and target for feature importance evaluation
    X = df_clean.drop(columns=['date', target_col])
    y = df_clean[target_col]
    
    # Train a quick Random Forest
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Extract top features
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.nlargest(top_n).index.tolist()
    print(f"Top {top_n} features selected: {top_features}")
    
    # Return the dataset with only the date, target, and top features
    final_columns = ['date', target_col] + top_features
    return df_clean[final_columns]

def main():
    input_path = '../data/processed/cleaned_data.csv'
    output_path = '../data/processed/train_ready_data.csv'
    
    df = load_processed_data(input_path)
    
    # Feature Generation
    df = create_time_features(df)
    df = create_lagged_features(df)
    df = create_rolling_features(df)
    df = create_interaction_features(df)
    
    # Feature Selection
    df_final = select_top_features(df, top_n=15)
    
    # Save Final Dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()