import pytest
import pandas as pd
import os
from src.power_forecaster import PowerForecaster

# Create mock data files for testing if they don't exist
@pytest.fixture(scope="session")
def setup_mock_inputs(tmp_path_factory):
    input_dir = tmp_path_factory.mktemp("mock_inputs")
    # Create a simple, consistent dataset for testing
    data = {
        'Time': pd.to_datetime(pd.date_range(start="2023-01-01", periods=100, freq='H')),
        'temperature_2m': range(100),
        'relativehumidity_2m': [50.0] * 100,
        'dewpoint_2m': [10.0] * 100,
        'windspeed_10m': range(100),
        'windspeed_100m': range(100, 200),
        'winddirection_10m': [180.0] * 100,
        'winddirection_100m': [185.0] * 100,
        'windgusts_10m': range(200, 300),
        'Power': [i % 100 / 100.0 for i in range(100)] # Power between 0 and 1
    }
    df = pd.DataFrame(data)
    
    for i in range(1, 5):
        file_path = input_dir / f'Location{i}.csv'
        df.to_csv(file_path, index=False)
        
    return str(input_dir)

@pytest.fixture
def forecaster_instance(setup_mock_inputs, tmp_path):
    # Use a temporary output directory for testing
    output_dir = tmp_path / "outputs"
    os.makedirs(output_dir, exist_ok=True)
    return PowerForecaster(data_dir=setup_mock_inputs, output_dir=str(output_dir))

def test_load_data(forecaster_instance):
    df = forecaster_instance.load_data(site_index=1)
    assert not df.empty
    assert 'Time' in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df['Time'])

def test_plot_timeseries(forecaster_instance):
    # This test mainly checks if the function runs without errors and creates a file
    forecaster_instance.plot_timeseries(
        variable_name='Power',
        site_index=1,
        starting_time='2023-01-01',
        ending_time='2023-01-05'
    )
    plot_path = os.path.join(forecaster_instance.output_dir, 'data_files_you_generate', 'timeseries_site1_Power.png')
    assert os.path.exists(plot_path)

def test_prepare_forecasting_data(forecaster_instance):
    X_train, X_test, y_train, y_test, times = forecaster_instance.prepare_forecasting_data(site_index=1)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert 'Power' in X_train.columns # Current power is a feature
    assert len(times) == len(y_test)
    assert X_test.shape[0] == 20 # 20% of 99 data points after dropping NaN

def test_train_and_evaluate_ml_model(forecaster_instance):
    # Test SVR
    forecaster_instance.train_model(site_index=1, model_name='SVR')
    metrics, _, _ = forecaster_instance.evaluate_model(site_index=1, model_name='SVR')
    assert len(metrics) == 3 # MSE, MAE, RMSE
    assert metrics[0] >= 0 # MSE should be non-negative

    # Test RandomForest
    forecaster_instance.train_model(site_index=1, model_name='RandomForest')
    metrics, _, _ = forecaster_instance.evaluate_model(site_index=1, model_name='RandomForest')
    assert len(metrics) == 3
    assert metrics[0] >= 0

def test_evaluate_persistence_model(forecaster_instance):
    metrics, _, _ = forecaster_instance.evaluate_persistence_model(site_index=1)
    assert len(metrics) == 3
    assert metrics[0] >= 0

def test_plot_predictions(forecaster_instance):
    # Ensure model is trained first
    forecaster_instance.train_model(site_index=1, model_name='SVR')
    
    forecaster_instance.plot_predictions(
        site_index=1,
        model_name='SVR',
        starting_time='2023-01-04 00:00:00',
        ending_time='2023-01-05 00:00:00'
    )
    plot_path = os.path.join(forecaster_instance.output_dir, 'data_files_you_generate', 'prediction_plot_site1_SVR.png')
    assert os.path.exists(plot_path)