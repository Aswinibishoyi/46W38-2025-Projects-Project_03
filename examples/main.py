import pandas as pd
import sys
import os
#update code
# Define project_root explicitly to ensure it's accessible and correct for path modification
project_root = 'D:/Project03_46W38'
# Get the path to the project root directory and add it to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)



# Assuming PowerForecaster is defined in a module named 'power_forecaster'
from src.power_forecaster import PowerForecaster , EDAAnalyzer


# This line was causing a SyntaxError. It appears to be an unassigned string.
# /My Drive/Colab Notebooks/46W38 Scientific Programming in Wind Energy/Proj2
inputs_folder = os.path.join(project_root, 'inputs')
output_folder = os.path.join(project_root, 'outputs')
examples_folder = os.path.join(project_root, 'examples')
source_folder = os.path.join(project_root, 'src')

print(f"Input folder created at: {inputs_folder}")
print(f"Output folder created at: {output_folder}")
print(f"Examples folder created at: {examples_folder}")
print(f"Output folder created at: {source_folder}")

# Add the project_root to the system path so Python can find the 'src' directory
# (Assuming the class code above is saved as src/power_forecaster.py based on original PDF)
# project_root and source_folder setup from original PDF is needed here

# For demonstration purposes, assuming the class code is available in the environment
# from src.power_forecaster import PowerForecaster, EDAAnalyzer

def create_comparison_table(metrics_dict):
    """Helper function to format evaluation metrics nicely."""
    data = {model: [m[0], m[1], m[2]] for model, m in metrics_dict.items()}
    df = pd.DataFrame(data, index=['MSE', 'MAE', 'RMSE']).T
    print("\n--- Model Comparison Table ---")
    print(df.to_markdown(floatfmt=".4f"))
    print("------------------------------\n")

def main_demo():
    print("Starting Project03_46W38 Demonstration...")
    # Initialize the forecaster class with correct absolute paths
    # (Update paths as necessary for your specific environment)
    forecaster = PowerForecaster(data_dir=inputs_folder, output_dir=output_folder)
    SITE_IDX = 3
    START_TIME = '2017-02-01 00:00:00'
    END_TIME = '2021-12-31 23:00:00'
    PLOT_START = '2020-11-01 00:00:00'
    PLOT_END = '2021-11-07 00:00:00'

    # --- Step 0 & 1 (EDA and Plotting) remain the same ---
    print(f"\n0. Performing EDA for Site {SITE_IDX}...")
    eda_analyzer = EDAAnalyzer(forecaster)
    eda_analyzer.analyze_site_data(SITE_IDX)

    print(f"\n1. Generating timeseries plots for Site {SITE_IDX}...")
    forecaster.plot_timeseries(variable_name='windspeed_100m', site_index=SITE_IDX, starting_time=START_TIME, ending_time=END_TIME)
    forecaster.plot_timeseries(variable_name='Power', site_index=SITE_IDX, starting_time=START_TIME, ending_time=END_TIME)

    # --- Step 2: Train all Models (SVR, RF, NN, and LSTM) ---
    print(f"\n2. Training all models for Site {SITE_IDX}...")
    models_to_train = ['SVR', 'RandomForest', 'NN', 'LSTM']
    for model_name in models_to_train:
      forecaster.train_model(site_index=SITE_IDX, model_name=model_name)

    # --- Step 3: Evaluate Models and Create Comparison Table ---
    print(f"\n3. Evaluating all models for Site {SITE_IDX}...")
    metrics = {}
    # Evaluate Persistence Model
    metrics['Persistence'], _, _ = forecaster.evaluate_persistence_model(site_index=SITE_IDX)
    # Evaluate ML/DL models
    for model_name in models_to_train:
        metrics[model_name], _, _ = forecaster.evaluate_model(site_index=SITE_IDX, model_name=model_name)

    create_comparison_table(metrics)

    # --- Step 4: Plot Predictions for a Specific Period ---
    print(f"""\n4. Generating prediction plots for Site {SITE_IDX} in the test period.""")
    # Plot predictions for all models
    for model_name in models_to_train:
        forecaster.plot_predictions(
            site_index=SITE_IDX,
            model_name=model_name,
            starting_time=PLOT_START,
            ending_time=PLOT_END
        )

    print("""\nDemonstration concluded successfully. Check the 'outputs' folder for results.""")

if __name__ == "__main__":
    # In a real setup, ensure inputs exist before running.
    # We run in "demo" mode which trains and saves models as needed.
    main_demo()