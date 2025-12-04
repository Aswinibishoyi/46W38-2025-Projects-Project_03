
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
import numpy as np

class PowerForecaster:
    """
    A class for loading, analyzing, and forecasting wind power data
    from multiple locations.
    """
    def __init__(self, data_dir='./inputs', output_dir='./outputs'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.models_dir = os.path.join(output_dir, 'trained_models')
        self.data_cache = {}
        # Ensure output directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data_files_you_generate'), exist_ok=True)

    def load_data(self, site_index: int) -> pd.DataFrame:
        """Loads data for a specific site."""
        if site_index in self.data_cache:
            return self.data_cache[site_index]

        file_path = os.path.join(self.data_dir, f'Location{site_index}.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")

        df = pd.read_csv(file_path)
        # Ensure Time column is treated as datetime for easy filtering
        df['Time'] = pd.to_datetime(df['Time'])
        self.data_cache[site_index] = df
        return df

    def plot_timeseries(self, variable_name: str, site_index: int, starting_time: str, ending_time: str):
        """
        Plots timeseries of a selected variable for a given site within a specific period.
        """
        df = self.load_data(site_index)
        start = pd.to_datetime(starting_time)
        end = pd.to_datetime(ending_time)

        filtered_df = df[(df['Time'] >= start) & (df['Time'] <= end)]

        if filtered_df.empty:
            print(f"No data found for the specified period.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(filtered_df['Time'], filtered_df[variable_name], label=f'Site {site_index} - {variable_name}')
        plt.xlabel("Time")
        plt.ylabel(variable_name)
        plt.title(f"Timeseries Plot: {variable_name} (Site {site_index})")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(self.output_dir, 'data_files_you_generate', f'timeseries_site{site_index}_{variable_name}.png')
        plt.savefig(plot_path)
        print(f"Saved plot to {plot_path}")
        plt.close()

    def prepare_forecasting_data(self, site_index: int):
        """
        Prepares data for ML forecasting by creating lag features.
        The input features (X) will be current conditions to predict
        the power output (y) in the next hour.
        """
        df = self.load_data(site_index).copy()

        # We assume data is hourly and sorted by time.
        # Shift 'Power' to get the target variable (next hour's power)
        df['Power_next_hour'] = df['Power'].shift(-1)

        # Drop the last row which will have NaN for the target
        df.dropna(inplace=True)

        # Define features (X) and target (y)
        # We use all available environmental data plus the current power output
        feature_columns = ['temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
                           'windspeed_10m', 'windspeed_100m', 'winddirection_10m',
                           'winddirection_100m', 'windgusts_10m', 'Power']

        X = df[feature_columns]
        y = df['Power_next_hour']

        # Split into training and test sets (e.g., first 80% train, last 20% test)
        split_index = int(len(df) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        return X_train, X_test, y_train, y_test, df.iloc[split_index:]['Time']

    def _get_metrics(self, y_true, y_pred):
        """Helper to compute metrics."""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return mse, mae, rmse

    def evaluate_persistence_model(self, site_index: int):
        """Calculates metrics for the persistence model (P_t+1 = P_t)."""
        # Need X_test to get 'Power' for persistence prediction
        _, X_test, _, y_test, _ = self.prepare_forecasting_data(site_index)
        # Persistence model assumes next hour power equals current hour power
        y_pred_persistence = X_test['Power']

        return self._get_metrics(y_test, y_pred_persistence), y_test, y_pred_persistence

    def train_model(self, site_index: int, model_name: str):
        """Trains an ML model and saves it."""
        # prepare_forecasting_data returns 5 values
        X_train, _, y_train, _, _ = self.prepare_forecasting_data(site_index)
        model_path = os.path.join(self.models_dir, f'{model_name}_site{site_index}.pkl')

        if model_name == 'SVR':
            model = SVR(C=1.0, epsilon=0.2)
        elif model_name == 'RandomForest':
            model = RandomForestRegressor(n_estimators=50, random_state=42)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        print(f"Training {model_name} for site {site_index}...")
        model.fit(X_train, y_train)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Trained model saved to {model_path}")
        return model

    def predict(self, site_index: int, model_name: str):
        """Loads a trained model and makes predictions on the test set."""
        _, X_test, _, y_test, test_times = self.prepare_forecasting_data(site_index)
        model_path = os.path.join(self.models_dir, f'{model_name}_site{site_index}.pkl')

        if not os.path.exists(model_path):
            print(f"Model not found. Training first...")
            self.train_model(site_index, model_name)

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        y_pred = model.predict(X_test)
        return y_test, y_pred, test_times

    def evaluate_model(self, site_index: int, model_name: str):
        """Calculates metrics for a trained ML model."""
        y_test, y_pred, _ = self.predict(site_index, model_name)
        return self._get_metrics(y_test, y_pred), y_test, y_pred

    def plot_predictions(self, site_index: int, model_name: str, starting_time: str, ending_time: str):
        """Plots predicted vs real power output for a given period."""
        y_true, y_pred, test_times = self.predict(site_index, model_name)

        # Combine into a temporary DF to filter by time
        results_df = pd.DataFrame({'Time': test_times, 'Actual Power': y_true, 'Predicted Power': y_pred})
        results_df = results_df.set_index('Time')

        start = pd.to_datetime(starting_time)
        end = pd.to_datetime(ending_time)
        filtered_df = results_df[(results_df.index >= start) & (results_df.index <= end)]

        if filtered_df.empty:
            print(f"No test data in the specified period for plotting.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(filtered_df.index, filtered_df['Actual Power'], label='Actual Power', color='blue', alpha=0.7)
        plt.plot(filtered_df.index, filtered_df['Predicted Power'], label=f'{model_name} Prediction', color='red', linestyle='--')
        plt.xlabel("Time")
        plt.ylabel("Normalized Power Output")
        plt.title(f"Actual vs. Predicted Power (Site {site_index}) using {model_name}")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(self.output_dir, 'data_files_you_generate', f'prediction_plot_site{site_index}_{model_name}.png')
        plt.savefig(plot_path)
        print(f"Saved prediction plot to {plot_path}")
        plt.close()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler # Added import
import pickle
import os
import numpy as np
# Import TensorFlow and Keras components
import tensorflow as tf # Added import
from tensorflow.keras.models import Sequential # Added import
from tensorflow.keras.layers import Dense, LSTM # Added import


class PowerForecaster:
    """
    A class for loading, analyzing, and forecasting wind power data
    from multiple locations.
    """
    def __init__(self, data_dir='./inputs', output_dir='./outputs'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.models_dir = os.path.join(output_dir, 'trained_models')
        self.data_cache = {}
        # Ensure output directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data_files_you_generate'), exist_ok=True)
        self.scalers = {} # Dictionary to store scalers for each site

    def load_data(self, site_index: int) -> pd.DataFrame:
        """Loads data for a specific site."""
        if site_index in self.data_cache:
            return self.data_cache[site_index]
        file_path = os.path.join(self.data_dir, f'Location{site_index}.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")
        df = pd.read_csv(file_path)
        # Ensure Time column is treated as datetime for easy filtering
        df['Time'] = pd.to_datetime(df['Time'])
        self.data_cache[site_index] = df
        return df

    def plot_timeseries(self, variable_name: str, site_index: int, starting_time: str, ending_time: str):
        """
        Plots timeseries of a selected variable for a given site within a specific
        period.
        """
        df = self.load_data(site_index)
        start = pd.to_datetime(starting_time)
        end = pd.to_datetime(ending_time)
        filtered_df = df[(df['Time'] >= start) & (df['Time'] <= end)]
        if filtered_df.empty:
            print(f"No data found for the specified period.")
            return
        plt.figure(figsize=(10, 6))
        plt.plot(filtered_df['Time'], filtered_df[variable_name], label=f'Site {site_index} - {variable_name}')
        plt.xlabel("Time")
        plt.ylabel(variable_name)
        plt.title(f"Timeseries Plot: {variable_name} (Site {site_index})")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.output_dir, 'data_files_you_generate', f'timeseries_site{site_index}_{variable_name}.png')
        plt.savefig(plot_path)
        print(f"Saved plot to {plot_path}")
        plt.close()

    def prepare_forecasting_data(self, site_index: int, model_type: str = 'sklearn'):
        """
        Prepares data for ML forecasting by creating lag features.
        The input features (X) will be current conditions to predict
        the power output (y) in the next hour.
        """
        df = self.load_data(site_index).copy()
        # We assume data is hourly and sorted by time.
        # Shift 'Power' to get the target variable (next hour's power)
        df['Power_next_hour'] = df['Power'].shift(-1)
        # Drop the last row which will have NaN for the target
        df.dropna(inplace=True)
        # Define features (X) and target (y)
        feature_columns = ['temperature_2m', 'relativehumidity_2m', 'dewpoint_2m',
                           'windspeed_10m', 'windspeed_100m', 'winddirection_10m',
                           'winddirection_100m', 'windgusts_10m', 'Power']
        X = df[feature_columns]
        y = df['Power_next_hour']

        # Split into training and test sets (e.g., first 80% train, last 20% test)
        split_index = int(len(df) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        test_times = df.iloc[split_index:]['Time']

        if model_type in ['NN', 'LSTM']:
            # Initialize separate scalers for features and target
            X_scaler = MinMaxScaler(feature_range=(0, 1))
            y_scaler = MinMaxScaler(feature_range=(0, 1))

            # Scale features
            X_train_scaled = X_scaler.fit_transform(X_train)
            X_test_scaled = X_scaler.transform(X_test)

            # Scale target (y needs to be reshaped for scaler)
            y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

            # Store both scalers separately for the site
            self.scalers[f'X_scaler_{site_index}'] = X_scaler
            self.scalers[f'y_scaler_{site_index}'] = y_scaler

            if model_type == 'LSTM':
                # Reshape data to 3D for LSTM: [samples, time steps, features]
                n_features = X_train_scaled.shape[1]
                X_train_scaled = np.array(X_train_scaled).reshape(-1, 1, n_features)
                X_test_scaled = np.array(X_test_scaled).reshape(-1, 1, n_features)

            X_train, X_test, y_train, y_test = X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

        return X_train, X_test, y_train, y_test, test_times

    def _get_metrics(self, y_true, y_pred):
        """Helper to compute metrics."""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return mse, mae, rmse

    def evaluate_persistence_model(self, site_index: int):
        """Calculates metrics for the persistence model (P_t+1 = P_t)."""
        # Need X_test to get 'Power' for persistence prediction
        _, X_test, _, y_test, _ = self.prepare_forecasting_data(site_index, model_type='sklearn')
        # Persistence model assumes next hour power equals current hour power
        y_pred_persistence = X_test['Power']
        return self._get_metrics(y_test, y_pred_persistence), y_test, y_pred_persistence

    def train_model(self, site_index: int, model_name: str):
        """Trains an ML model and saves it."""
        model_type = 'sklearn'
        if model_name in ['NN', 'LSTM']:
            model_type = model_name

        # prepare_forecasting_data returns 5 values
        X_train, _, y_train, _, _ = self.prepare_forecasting_data(site_index, model_type=model_type)


        model = None
        if model_name == 'SVR':
            model = SVR(C=1.0, epsilon=0.2)
            model_path = os.path.join(self.models_dir, f'{model_name}_site{site_index}.pkl')
        elif model_name == 'RandomForest':
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model_path = os.path.join(self.models_dir, f'{model_name}_site{site_index}.pkl')
        elif model_name == 'NN':
            model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model_path = os.path.join(self.models_dir, f'{model_name}_site{site_index}.keras') # Added .keras extension
        elif model_name == 'LSTM':
            # Input shape for LSTM is [samples, time steps, features]
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model_path = os.path.join(self.models_dir, f'{model_name}_site{site_index}.keras') # Added .keras extension
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        print(f"Training {model_name} for site {site_index}...")

        if model_name in ['SVR', 'RandomForest']:
             model.fit(X_train, y_train)
             with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        elif model_name in ['NN', 'LSTM']:
            # Fit Keras model, no pickle dump for Keras models
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            model.save(model_path)

        print(f"Trained model saved to {model_path}")
        return model

    def predict(self, site_index: int, model_name: str):
        """Loads a trained model and makes predictions on the test set."""
        model_type = 'sklearn'
        if model_name in ['NN', 'LSTM']:
            model_type = model_name

        _, X_test, _, y_test, test_times = self.prepare_forecasting_data(site_index, model_type=model_type)

        if model_name in ['SVR', 'RandomForest']:
            model_path = os.path.join(self.models_dir, f'{model_name}_site{site_index}.pkl')
            if not os.path.exists(model_path):
                print(f"Model not found. Training first...")
                self.train_model(site_index, model_name)
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            y_pred = model.predict(X_test)
        elif model_name in ['NN', 'LSTM']:
            model_path = os.path.join(self.models_dir, f'{model_name}_site{site_index}.keras') # Added .keras extension
            if not os.path.exists(model_path):
                 print(f"Model not found. Training first...")
                 self.train_model(site_index, model_name)
            model = tf.keras.models.load_model(model_path)
            y_pred_scaled = model.predict(X_test).flatten()

            # Inverse transform predictions to original scale using the y_scaler
            y_scaler = self.scalers[f'y_scaler_{site_index}']
            # Need to reshape the scaled predictions back for inverse_transform
            y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten() # y_test is already scaled and flat, needs reshaping for inverse_transform

        return y_test, y_pred, test_times

    def evaluate_model(self, site_index: int, model_name: str):
        """Calculates metrics for a trained ML model."""
        y_test, y_pred, _ = self.predict(site_index, model_name)
        return self._get_metrics(y_test, y_pred), y_test, y_pred

    def plot_predictions(self, site_index: int, model_name: str, starting_time: str, ending_time: str):
        """Plots predicted vs real power output for a given period."""
        y_true, y_pred, test_times = self.predict(site_index, model_name)
        # Combine into a temporary DF to filter by time
        results_df = pd.DataFrame({'Time': test_times, 'Actual Power': y_true, 'Predicted Power': y_pred})
        results_df = results_df.set_index('Time')
        start = pd.to_datetime(starting_time)
        end = pd.to_datetime(ending_time)
        filtered_df = results_df[(results_df.index >= start) & (results_df.index <= end)]
        if filtered_df.empty:
            print(f"No test data in the specified period for plotting.")
            return
        plt.figure(figsize=(12, 6))
        plt.plot(filtered_df.index, filtered_df['Actual Power'], label='Actual Power', color='blue', alpha=0.7)
        plt.plot(filtered_df.index, filtered_df['Predicted Power'], label=f'{model_name} Prediction', color='red', linestyle='--')
        plt.xlabel("Time")
        plt.ylabel("Normalized Power Output")
        plt.title(f"Actual vs. Predicted Power (Site {site_index}) using {model_name}")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.output_dir, 'data_files_you_generate', f'prediction_plot_site{site_index}_{model_name}.png')
        plt.savefig(plot_path)
        print(f"Saved prediction plot to {plot_path}")
        plt.close()


class EDAAnalyzer:
    """
    A class for performing exploratory data analysis on the wind power data.
    """
    def __init__(self, forecaster):
        self.forecaster = forecaster
        self.output_dir = forecaster.output_dir

    def analyze_site_data(self, site_index: int):
        """Performs EDA for a specific site."""
        df = self.forecaster.load_data(site_index)
        print(f"\n--- Starting EDA for Site {site_index} ---")

        # Display basic information and summary statistics
        print("\nHead of the DataFrame:")
        print(df.head().to_markdown())
        print("\nDataFrame Info:")
        df.info()
        print("\nSummary Statistics:")
        print(df.describe().to_markdown(floatfmt=".2f"))

        # Plot correlation matrix heatmap
        self.plot_correlation_heatmap(df, site_index)

        print(f"--- EDA for Site {site_index} Concluded ---")

    def plot_correlation_heatmap(self, df: pd.DataFrame, site_index: int):
        """Generates and saves a correlation heatmap."""
        # Calculate correlation matrix
        correlation_matrix = df.corr(numeric_only=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f"Correlation Matrix Heatmap (Site {site_index})")

        # Save the plot
        plot_path = os.path.join(self.output_dir, 'data_files_you_generate', f'correlation_heatmap_site{site_index}.png')
        plt.savefig(plot_path)
        print(f"Saved correlation heatmap to {plot_path}")
        plt.close()