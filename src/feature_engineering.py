import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

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
    Creates rolling window features (mean, std, min, max) for specified window sizes.
    Default windows=[6, 18] assume a 10-minute sampling rate (6 -> 1 hour, 18 -> 3 hours).
    """
    df_feat = df.copy()
    for window in windows:
        rol = df_feat[target_col].rolling(window=window)
        df_feat[f'{target_col}_rolling_mean_{window}'] = rol.mean()
        df_feat[f'{target_col}_rolling_std_{window}'] = rol.std()
        df_feat[f'{target_col}_rolling_min_{window}'] = rol.min()
        df_feat[f'{target_col}_rolling_max_{window}'] = rol.max()
        df_feat[f'{target_col}_rolling_median_{window}'] = rol.median()
    return df_feat

def create_lagged_features(df, target_col='Appliances', lags=[1, 3, 6]):
    """
    Creates lag features for previous time steps. Lags are expressed in number of rows.
    e.g., with 10-minute frequency, lag=1 -> 10 minutes ago.
    """
    df_feat = df.copy()
    for lag in sorted(set(lags)):
        df_feat[f'{target_col}_lag_{lag}'] = df_feat[target_col].shift(lag)
    return df_feat


def determine_optimal_lags(series, max_lag=72, threshold=0.2):
    """
    Computes autocorrelation for lags 1..max_lag and returns lags with
    autocorrelation magnitude above `threshold`. Useful to pick candidate lags.
    Returns a list of suggested integer lags ordered by descending autocorrelation.
    """
    acorrs = {}
    for lag in range(1, max_lag + 1):
        try:
            ac = series.autocorr(lag=lag)
        except Exception:
            ac = np.nan
        acorrs[lag] = ac
    ac_series = pd.Series(acorrs).dropna()
    # rank by absolute correlation and filter by threshold
    strong = ac_series[ac_series.abs() >= threshold].abs().sort_values(ascending=False)
    suggested = strong.index.tolist()
    return suggested

def create_interaction_features(df):
    """Creates interaction terms between key environmental features."""
    df_feat = df.copy()
    if 'T_out' in df_feat.columns and 'RH_out' in df_feat.columns:
        df_feat['T_out_RH_out_interact'] = df_feat['T_out'] * df_feat['RH_out']
    # additional interactions commonly useful for energy prediction
    if 'T_out' in df_feat.columns and 'Appliances' in df_feat.columns:
        df_feat['T_out_Appliances_interact'] = df_feat['T_out'] * df_feat['Appliances']
    if 'RH_out' in df_feat.columns and 'Appliances' in df_feat.columns:
        df_feat['RH_out_Appliances_interact'] = df_feat['RH_out'] * df_feat['Appliances']
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


def feature_selection_rfe(df, target_col='Appliances', n_features_to_select=15):
    """
    Uses Recursive Feature Elimination (RFE) with a Random Forest estimator to
    select a given number of features. Returns the cleaned dataframe with
    selected features and a short report dict.
    """
    df_clean = df.dropna().copy()
    X = df_clean.drop(columns=['date', target_col])
    y = df_clean[target_col]

    estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=0.1)
    selector = selector.fit(X, y)

    selected_cols = X.columns[selector.support_].tolist()
    report = {
        'selected_features': selected_cols,
        'ranking': dict(zip(X.columns, selector.ranking_))
    }
    final_columns = ['date', target_col] + selected_cols
    return df_clean[final_columns], report


def summarize_selection(df, selected_columns, rf_model=None):
    """
    Prints a brief justification for selected features using correlation and
    optional tree-based importances when `rf_model` is provided.
    """
    print('Feature selection justification:')
    dfc = df.dropna().copy()
    corr = dfc[selected_columns + ['Appliances']].corr()['Appliances'].abs().sort_values(ascending=False)
    print('- Top correlations with target:')
    print(corr.head(10))
    if rf_model is not None:
        try:
            X = dfc[selected_columns]
            importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            print('- Tree-based importances:')
            print(importances.head(10))
        except Exception:
            pass

def main():
    input_path = '../data/processed/cleaned_data.csv'
    output_path = '../data/processed/train_ready_data.csv'
    
    df = load_processed_data(input_path)
    
    # Feature Generation
    df = create_time_features(df)

    # Determine useful lag candidates via autocorrelation
    suggested_lags = determine_optimal_lags(df['Appliances'], max_lag=72, threshold=0.2)
    if len(suggested_lags) == 0:
        # fallback defaults (1, 3, 6, 18 correspond to 10m,30m,1h,3h)
        suggested_lags = [1, 3, 6, 18]

    df = create_lagged_features(df, lags=suggested_lags)
    df = create_rolling_features(df, windows=[6, 18])
    df = create_interaction_features(df)

    # Prepare cleaned dataset for selection
    df_clean = df.dropna().copy()

    # Train a Random Forest for importance-based selection and for summaries
    X = df_clean.drop(columns=['date', 'Appliances'])
    y = df_clean['Appliances']
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Tree-based top features
    df_tree_selected = select_top_features(df, target_col='Appliances', top_n=15)

    # RFE-based selection
    df_rfe, rfe_report = feature_selection_rfe(df, target_col='Appliances', n_features_to_select=15)

    # Summarize justifications
    summarize_selection(df, rfe_report['selected_features'], rf_model=rf)

    # Save the RFE-selected dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_rfe.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()