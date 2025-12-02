import pandas as pd
from src.power_forecaster import PowerForecaster

def create_comparison_table(metrics_dict):
    """Helper function to format evaluation metrics nicely."""
    data = {model: [m[0], m[1], m[2]] for model, m in metrics_dict.items()}
    df = pd.DataFrame(data, index=['MSE', 'MAE', 'RMSE']).T
    print("\n--- Model Comparison Table ---")
    print(df.to_markdown(floatfmt=".4f"))
    print("------------------------------\n")


def main_demo():
    print("Starting Project03_46W38 Demonstration...")
    
    # Initialize the forecaster class
    forecaster = PowerForecaster(data_dir='./inputs', output_dir='./outputs')
    
    SITE_IDX = 1
    START_TIME = '2023-01-01 00:00:00'
    END_TIME = '2023-01-07 23:00:00'

    # --- Step 1: Load and Plot Time Series ---
    print(f"\n1. Generating timeseries plots for Site {SITE_IDX}...")
    forecaster.plot_timeseries(
        variable_name='windspeed_100m', 
        site_index=SITE_IDX, 
        starting_time=START_TIME, 
        ending_time=END_TIME
    )
    forecaster.plot_timeseries(
        variable_name='Power', 
        site_index=SITE_IDX, 
        starting_time=START_TIME, 
        ending_time=END_TIME
    )

    # --- Step 2: Train Models (SVR and RandomForest) ---
    # These models will be saved to outputs/trained_models
    print(f"\n2. Training ML models for Site {SITE_IDX}...")
    forecaster.train_model(site_index=SITE_IDX, model_name='SVR')
    forecaster.train_model(site_index=SITE_IDX, model_name='RandomForest')

    # --- Step 3: Evaluate Models and Create Comparison Table ---
    print(f"\n3. Evaluating persistence model and ML models for Site {SITE_IDX}...")
    
    metrics = {}
    
    # Evaluate Persistence Model
    metrics['Persistence'], _, _ = forecaster.evaluate_persistence_model(site_index=SITE_IDX)
    
    # Evaluate SVR
    metrics['SVR'], _, _ = forecaster.evaluate_model(site_index=SITE_IDX, model_name='SVR')

    # Evaluate RandomForest
    metrics['RandomForest'], _, _ = forecaster.evaluate_model(site_index=SITE_IDX, model_name='RandomForest')

    create_comparison_table(metrics)

    # --- Step 4: Plot Predictions for a Specific Period ---
    print(f"\n4. Generating prediction plots for Site {SITE_IDX} in the test period...")
    # Using a shorter specific period within the test set for clear visualization
    PLOT_START = '2023-11-01 00:00:00'
    PLOT_END = '2023-11-07 00:00:00'

    forecaster.plot_predictions(
        site_index=SITE_IDX,
        model_name='RandomForest',
        starting_time=PLOT_START,
        ending_time=PLOT_END
    )
    
    forecaster.plot_predictions(
        site_index=SITE_IDX,
        model_name='SVR',
        starting_time=PLOT_START,
        ending_time=PLOT_END
    )

    print("\nDemonstration concluded successfully. Check the 'outputs' folder for results.")

if __name__ == "__main__":
    # In a real setup, ensure inputs exist before running.
    # We run in "demo" mode which trains and saves models as needed.
    main_demo()































