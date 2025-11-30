import pandas as pd
import os

# Import functions from your source package
from src.processing import load_and_process_data, evaluate_persistence_model

# --- Configuration Variables ---
# Assumes data files are in Project03_46W38/inputs/
# You will need to create the 'inputs' directory and add dummy data files for this to run.
folder_path = os.path.join(os.getcwd(), 'inputs') 
file_pattern = 'location_data_{}.csv'
num_locations = 4
locations_to_process = [f'location-{i}' for i in range(1, num_locations + 1)]

# --- Main Workflow ---

def main():
    print("--- Starting Data Processing and Evaluation ---")
    
    # 1. Load and process data
    all_data = load_and_process_data(folder_path, file_pattern, num_locations)
    
    if not all_data.empty:
        print("\nData successfully loaded. Preview:")
        print(all_data.head())
        print(all_data.info())
        
        # 2. Run evaluation for all locations
        results_list = []
        for location_id in locations_to_process:
            metrics = evaluate_persistence_model(all_data, location_id)
            if metrics:
                results_list.append(metrics)
        
        # 3. Tabulate results
        if results_list:
            performance_table = pd.DataFrame(results_list)
            print("\n--- Performance Metrics Tabulated for All Locations (Persistence Model) ---")
            # Assumes output directory exists
            output_path = os.path.join(os.getcwd(), 'outputs', 'performance_table.md')
            # Ensure the output directory exists before writing
            os.makedirs(os.path.join(os.getcwd(), 'outputs'), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(performance_table.to_markdown(index=False))
            print(f"Results saved to {output_path}")
        else:
            print("No evaluation results to display.")
    else:
        print("Could not load data. Exiting main function.")

if __name__ == "__main__":
    # This block allows the script to be run directly
    main()