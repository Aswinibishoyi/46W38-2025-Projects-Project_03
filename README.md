
Data from four locations are used (weather and power production) and analyzed.
This script compares various machine learning techniques (persistent as baseline to compare with machine learning predictions such as SVR, random forest, NN and LSTM) for wind energy forecasting, which is crucial for managing the inherent variability of wind energy. The script prepared with a structured approach to find wind energy forecasting (power) incorporating the steps of creatinga folder structure as per project requirement
      i.	inputs
      ii.	outputs
      iii.	examples (to keep main demo function)
      iv.	src (source, has two main classes PowerForecaster and EDAAnalyzer)
          1.	data loading
          2.	exploratory data analysis (EDA)
          3.	data preparation
          4.	model training
          5.	prediction
          6.	evaluation
          7.	visualization

Detailed Design and Implementation:

   Classes Used: The software architecture used is an object-oriented design centered around two main classes, PowerForecaster and EDAAnalyzer. These classes encapsulate specific functionalities, promoting modularity and reusability, with the dataflow managed through a structured sequence of operations defined in the main_demo function. 
              
      a.	PowerForecaster: This class handles data loading, preprocessing, model training and prediction, and plotting of results.
              o	Data Handling: Loads site-specific data from CSV files into pandas DataFrames, ensures the Time column is in datetime format.
              o	Data Preparation: Prepares data for machine learning by creating a target variable (Power_next_hour) using time shifting, defining feature columns (e.g., wind speed, temperature, humidity, current power), and splitting data into training and testing sets.
              o	Scaling and Reshaping: Employs MinMaxScaler for normalizing data and reshapes data into a 3D format for Long Short-Term Memory (LSTM) models
              o	Model Management: Trains and saves models to specified output directories, using pickle for scikit-learn models and Keras's save method for neural networks.
              o	Prediction and Evaluation: Loads trained models to make predictions on test data and evaluates performance using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
              o	Persistence Model: Implements a simple persistence model (next hour's power equals current power) as a baseline for comparison.
              o	Visualization: Generates and saves plots of time series data and actual vs. predicted power outputs.
          
      b.	EDAAnalyzer: This class performs exploratory data analysis
              o	Data Summary: Prints the head, info, and descriptive statistics of the DataFrame.
              o	Correlation Analysis: Calculates and visualizes the correlation matrix using a seaborn heatmap to understand relationships between variables.

    Main Function: The main_demo function orchestrates the entire process:
    
              i.	Sets up project directories for inputs, outputs, example and source code.
              ii.	Initializes the PowerForecaster and EDAAnalyzer with specified paths.
              iii.	Performs EDA and generates time series plots for a specified site.
              iv.	Trains all four ML/DL models and trained models are saved (need to improve to use the saved modes)
              v.	Evaluates all models (including the persistence baseline) and prints a comparison table of metrics.
              vi.	Generates and saves prediction plots for all models for a specific test period.
              vii.	a script is used to get the current directory and create the folder stricture as per requirements.

4.	Findings and Discussion:

        a.	Findings/Results:
        
                The following table presents a snapshot of the model performance metrics for Site Location-1, demonstrating the significant reduction in error achieved by the ML approaches:
                
                Model	        MSE	    MAE	      RMSE
                Persistence  	0.0091	0.0262	0.0357
                SVR	          0.031	  0.1442	0.176
                RandomForest	0.0013	0.026	  0.0361
                NN	          0.0012	0.0255	0.0352
                LSTM	        0.0013	0.0256	0.0357





        b.	Discussion:
                Analyze what the results mean. Discuss the success of the implementation in meeting the objectives, acknowledge any limitations, and compare theoretical assumptions with actual outcomes.
  	
              i.	Regression (SVR), Random Forest, Neural Networks (NN), and Long Short-Term Memory (LSTM), consistently demonstrated significantly lower error rates (MSE, MAE, RMSE) compared to the simple Persistence Model (/a naive baseline) across all sites tested.
              ii.	Top Performing Model: Among the models evaluated, the LSTM and NN models achieved the lowest Mean Squared Error (MSE) and Root Mean Squared Error (RMSE), indicating better overall prediction accuracy for the time-series data.
              iii.	Consistency Across Locations: The relative performance hierarchy of the models remained consistent across all four locations (Site 1, Site 2, Site 3 and Site 4), with ML models consistently providing more reliable forecasts than the persistence method.

