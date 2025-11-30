import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_and_process_data(folder_path, file_pattern, num_locations):
    """
    Loads data from multiple CSV files, adds a Location_ID, and concatenates them.
    """
    all_data = pd.DataFrame()
    for i in range(1, num_locations + 1):
        file_path = os.path.join(folder_path, file_pattern.format(i))
        try:
            df = pd.read_csv(file_path)
            # Modified line to use 'location-1', 'location-2', etc.
            df['Location_ID'] = f'location-{i}'
            all_data = pd.concat([all_data, df], ignore_index=True)
        except FileNotFoundError:
            print(f"Error: {file_path} not found.")
    return all_data

def evaluate_persistence_model(df, site_id_str):
    """
    Splits data for a given site, applies a persistence model,
    and computes MAE, MSE, and RMSE, returning metrics as a dictionary.
    """
    site_data = df[df['Location_ID'] == site_id_str].copy()
    if site_data.empty:
        print(f"Error: No data found for {site_id_str}. Check Location_ID naming.")
        return None
    
    site_data = site_data.sort_values(by='Time').reset_index(drop=True)
    train_size = int(len(site_data) * 0.8)
    test_data = site_data.iloc[train_size:]
    y_true = test_data['Power'].values
    last_train_power = site_data.iloc[train_size - 1]['Power']
    predictions = test_data['Power'].shift(1)
    predictions.iloc[0] = last_train_power
    y_pred = predictions.values
    
    # Compute error metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {"Location": site_id_str, "MAE": mae, "MSE": mse, "RMSE": rmse}